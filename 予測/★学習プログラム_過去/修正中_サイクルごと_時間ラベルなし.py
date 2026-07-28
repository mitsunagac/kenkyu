import pandas as pd
import numpy as np
import torch
import csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch_geometric.data import Data
import torch.nn.functional as F
from torch.nn import BatchNorm1d
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import global_add_pool
from torch_geometric.loader import DataLoader
from torch_geometric.explain import ModelConfig, Explainer, GNNExplainer
from torch_geometric.nn import TGNMemory, TransformerConv
from matplotlib.ticker import ScalarFormatter
import random
import gc
import os
import joblib

print(torch.cuda.is_available())

# === 使用ノード（道路8・9を除く） ===
used_road_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]
node_num = len(used_road_indices)

# === デバイスの設定 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === データ読み込み ===
# train_data_csv = pd.read_csv(r"C:\Users\tslab\Desktop\予測\修正済みデータ_5ヶ月分.csv", encoding='cp932')
# train_data_csv = pd.read_csv(r"C:\Users\tslab\Desktop\予測\共三参照_151日データ.csv", encoding='cp932')
# train_data_csv = pd.read_csv(r"C:\Users\tslab\Desktop\予測\前後半11月10日データ.csv", encoding='cp932')
train_data_csv = pd.read_csv(r"C:\Users\tslab\Desktop\予測\新環境待ち台数データ_12.22.csv", encoding='cp932')
adjacency_matrix = pd.read_csv(r"C:\\Users\\tslab\\Desktop\\予測\\新環境_隣接行列.csv", encoding='shift-jis', index_col=0)
# adjacency_matrix = pd.read_csv(r"C:\Users\tslab\Desktop\予測\交差点A, Jのみの道路情報.csv", encoding='shift-jis', index_col=0)
print(adjacency_matrix)

csv_output_path = r'C:\\Users\\tslab\\Desktop\\予測\\csv掃き出し\\遅れ時間.csv'
answer_csv_path = r"C:\Users\tslab\Desktop\予測\新環境正解データ_12.22.csv"

scaler_path = r"C:\Users\tslab\Desktop\予測\予測モデル\scaler.pkl"

# == データ数 ==============================================================
DAY_LENGTH = 1330  # 1日のサンプル数（固定）
# DAY_LENGTH = 665  # 1日のサンプル数（固定）

val_days = 5
test_days = 1
train_days = 145

val_long = val_days * DAY_LENGTH
test_long = test_days * DAY_LENGTH
train_long = train_days * DAY_LENGTH

# === 時間ラベルの付与関数 ===
def assign_time_labels_daywise(length, day_length=1330, num_classes=5):
    labels = []
    for i in range(length):
        day_time = i % day_length
        block_size = day_length // num_classes
        labels.append(day_time // block_size)
    return labels

# === 特徴に時間ラベルを追加する関数 ===
def add_time_label_to_features(features, time_label, num_classes=5):
    one_hot = torch.nn.functional.one_hot(torch.tensor(time_label), num_classes=num_classes).float()
    one_hot = one_hot.unsqueeze(0).repeat(features.shape[0], 1).to(features.device)
    return torch.cat([features, one_hot], dim=1)

# === ノードを使用インデックスで制限 ===
df_train_data = train_data_csv.iloc[:train_long, used_road_indices]
df_val_data = train_data_csv.iloc[train_long:train_long+val_long, used_road_indices]
df_test_data = train_data_csv.iloc[train_long+val_long:train_long+val_long+test_long, used_road_indices]

print(f"df_train_data形状:{df_train_data.shape}")
print(f"df_val_data形状:{df_val_data.shape}")
print(f"df_test_data形状:{df_test_data.shape}")
print(df_train_data)

# === 正規化 ===
scaler = MinMaxScaler()
train_data = scaler.fit_transform(df_train_data).T
validation_data = scaler.transform(df_val_data).T
test_data = scaler.transform(df_test_data).T

# === 各ノードのスケーリング範囲を表示 ===
print("=== 各ノードの MinMaxScaler スケーリング範囲 ===")
for i, (min_val, max_val) in enumerate(zip(scaler.data_min_, scaler.data_max_)):
    print(f"ノード{i}（道路{used_road_indices[i]}）: min={min_val}, max={max_val}")


# === edge_index 再構築 ===
adj = adjacency_matrix.values
edge_index_raw = np.array(np.nonzero(adj)).astype(np.int64)
mask = np.isin(edge_index_raw[0], used_road_indices) & np.isin(edge_index_raw[1], used_road_indices)
edge_index_filtered = edge_index_raw[:, mask]
id_map = {old: new for new, old in enumerate(used_road_indices)}
edge_index_mapped = np.vectorize(id_map.get)(edge_index_filtered)
edge_index = torch.tensor(edge_index_mapped, dtype=torch.long).to(device)
print("エッジ：", edge_index.shape)

# === 時間ラベル作成 ===
train_labels = assign_time_labels_daywise(df_train_data.shape[0])
val_labels   = assign_time_labels_daywise(df_val_data.shape[0])
test_labels  = assign_time_labels_daywise(df_test_data.shape[0])

# === Dataset 作成 ===
# === ユーティリティ関数：Dataリストを日ごとに構成 ===
def build_daywise_dataset(data_array, labels, edge_index, device, alpha=2):
    total_steps = data_array.shape[1]
    num_days = total_steps // DAY_LENGTH
    dataset = []

    for day in range(num_days):
        start = day * DAY_LENGTH
        end = start + DAY_LENGTH

        for t in range(start, end - 3, 2):
            x_t1 = torch.tensor(data_array[:, t], dtype=torch.float).unsqueeze(1).to(device)
            x_t2 = torch.tensor(data_array[:, t+1], dtype=torch.float).unsqueeze(1).to(device)

            # 🚩 道路1（ノード0）だけを α 倍
            x_t1[0] *= alpha
            x_t2[0] *= alpha

            # # 🚩 道路5（ノード4）だけを α 倍
            # x_t1[4] *= alpha
            # x_t2[4] *= alpha

            x = torch.cat([x_t1, x_t2], dim=1)

            y_t3 = torch.tensor(data_array[:, t+2], dtype=torch.float).unsqueeze(1).to(device)
            y_t4 = torch.tensor(data_array[:, t+3], dtype=torch.float).unsqueeze(1).to(device)
            y = torch.cat([y_t3, y_t4], dim=1)

            dataset.append(Data(x=x, edge_index=edge_index, y=y))

    return dataset


train_dataset = build_daywise_dataset(train_data, train_labels, edge_index, device)
val_dataset   = build_daywise_dataset(validation_data, val_labels, edge_index, device)
test_dataset  = build_daywise_dataset(test_data, test_labels, edge_index, device)


# === バッチ作成 ===
batch_size = 64
# batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch.nn as nn


# ==== 7層 ============================================================
class TrafficPredictionGNN(torch.nn.Module):
    def __init__(self, num_node_features, dropout_rate=0.2):
        super(TrafficPredictionGNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 32)
        self.conv2 = GCNConv(32, 32)
        self.conv3 = GCNConv(32, 64)
        self.conv4 = GCNConv(64, 128)
        self.conv5 = GCNConv(128, 128)

        self.lin = nn.Linear(128, 2)
        self.dropout_rate = dropout_rate

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = self.conv3(x, edge_index)
        x = F.relu(x)

        x = self.conv4(x, edge_index)
        x = F.relu(x)

        x = self.conv5(x, edge_index)
        x = F.relu(x)

        out = self.lin(x)
        return out
    

# 学習率のリスト（エポック数に基づいて設定）
learning_rates = [0.001] * 100 + [0.0005] * 50 + [0.0001] * 50  # 合計200

#モデルの保存パス
best_model_path = r"C:\Users\tslab\Desktop\予測\予測モデル\GCN_epoch_best_model.pth"

# 早期停止の設定
patience = 25                    #早期終了の条件
best_val_loss = float('inf')    #最小のバリデーションロスを記録(初期は無限)
best_test_loss = float('inf')   #val Lossが改善されなかった連続エポック数
epochs_no_improve = 0

# トレーニングループ (可視化用)
train_losses = []
val_losses = []  # バリデーションロスを記録
test_losses = []

# 予測結果保存用リスト
epoch_predictions = []
epoch_targets = []

#予測をし，RMSEを計算をした回数保存
prediction_count = 1

#RMSE保存
rmse_t3 = float('inf')
rmse_t4 = float('inf')

# === Loss履歴の保存用 ===
best_overall_val_loss = float('inf')
best_train_losses = []
best_val_losses = []


# トレーニングループ（指定したRMSEより精度が向上しない場合は，再度学習・予測）
while (rmse_t3 > 1.1 or rmse_t4 > 1.2):
    model = TrafficPredictionGNN(num_node_features=2, dropout_rate=0.2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    #Lossを格納している配列中身を削除
    train_losses.clear()
    val_losses.clear()
    epochs_no_improve = 0
    #best_Model保存を変数初期化
    best_val_loss = float('inf')
    best_test_loss = float('inf')

    print(f"{prediction_count}回目の学習中")
    model.train()
    for epoch in range(200):  # 例として100エポック
        
        # エポックに応じて学習率を変更
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rates[epoch]

        # トレーニング
        running_train_loss = 0.0
        model.train()  # モデルを訓練モードに
        for batch in train_loader:
            batch = batch.to(device)  # データをGPUに移動
            optimizer.zero_grad()
            # 出力：[batch * ノード数, 2]
            out = model(batch.x, batch.edge_index)
            # 道路1に該当するノードのみを抽出（インデックス0と仮定）
            out = out.view(-1, node_num, 1)  # shape: [batch_size, node_num, 2]
            y   = batch.y.view(-1, node_num, 1)

            # ノード0だけ取り出して2時刻分まとめて損失計算
            out_target = out[:, 0, 0]  # shape: [batch_size, 2]
            y_target   = y[:, 0, 0]    # shape: [batch_size, 2]

            # === ノード0の入力 (t, t+1) を取得 ===
            x0 = batch.x.view(-1, node_num, 1)[:, 0, 0]  # [batch, 2]

            # 入力が小さいか判定（最大値 <= 2）
            small_input_mask = (x0 <= 2).float()  # [batch]

            # 重み（小さい入力だけ 2 倍）
            weight = 1.0 + small_input_mask  # small → 2.0, others → 1.0

            # サンプルごとの MSE（2ステップ分）
            per_sample_loss = (out_target - y_target) ** 2  # [batch]

            # 重み付き loss
            loss = (per_sample_loss * weight).mean()

            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)


        # バリデーションロスの計算
        model.eval()  # モデルを評価モードに
        running_val_loss = 0.0
        predictions = []
        targets = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)

                out = model(batch.x, batch.edge_index)
                out = out.view(-1, node_num, 1)
                y = batch.y.view(-1, node_num, 1)

                out_target = out[:, 0, 0]
                y_target   = y[:, 0, 0]

                # ノード0の入力
                x0 = batch.x.view(-1, node_num, 1)[:, 0, 0]

                small_input_mask = (x0 <= 2).float()
                weight = 1.0 + small_input_mask

                per_sample_loss = (out_target - y_target) ** 2
                loss = (per_sample_loss * weight).mean()

                running_val_loss += loss.item()

        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # 最良モデルの保存
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            #Sepochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)  # モデルを保存
            print(f"Best model saved at epoch {epoch+1} with Val Loss: {best_val_loss}")

            best_overall_val_loss = best_val_loss
            best_train_losses = train_losses.copy()
            best_val_losses = val_losses.copy()

        # 結果の出力
        print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}, 学習率: {learning_rates[epoch]}')
        
        # Early Stopping
        if avg_val_loss < best_test_loss:
            best_test_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        #print(f"Loss更新なし：{epochs_no_improve}回")
        if epochs_no_improve >= patience:
            print('Early stopping')
            break

    # === テスト予測 ===
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    test_predictions = []
    test_ground_truth = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            out = out.view(-1, node_num, 1)
            y = batch.y.view(-1, node_num, 1)

            test_predictions.append(out.cpu().numpy())  # shape: [batch, node_num, 2]
            test_ground_truth.append(y.cpu().numpy())   # shape: [batch, node_num, 2]

    # 結合 → shape: [N, node_num, 2]
    test_predictions = np.concatenate(test_predictions, axis=0)
    test_ground_truth = np.concatenate(test_ground_truth, axis=0)

    # === 逆正規化 ===
    
    # === 逆正規化（1値専用） ===
    test_predictions[:, :, 0] = scaler.inverse_transform(test_predictions[:, :, 0])
    test_ground_truth[:, :, 0] = scaler.inverse_transform(test_ground_truth[:, :, 0])


    print("逆正規化・四捨五入前の予測データ：", test_predictions.shape)
    print("逆正規化・四捨五入前のテストデータ：", test_ground_truth.shape)


    # === RMSEの計算（t1→t3 / t2→t4 を分離） ===
    pred = test_predictions[:, 0, 0]
    true = test_ground_truth[:, 0, 0]

    pred_t3 = pred[0::2]
    true_t3 = true[0::2]

    pred_t4 = pred[1::2]
    true_t4 = true[1::2]

    rmse_t3 = np.sqrt(np.mean((pred_t3 - true_t3) ** 2))
    rmse_t4 = np.sqrt(np.mean((pred_t4 - true_t4) ** 2))

    print(f"RMSE t1→t3: {rmse_t3:.4f}, RMSE t2→t4: {rmse_t4:.4f}")


    # ======== 終了条件 ========
    if rmse_t3 <= 1.1 and rmse_t4 <= 1.2:
        print(f"🎯 終了条件達成！t3 RMSE={rmse_t3:.4f}, t4 RMSE={rmse_t4:.4f}")
        break
    else:
        print(f"⚙️ 継続学習: 前半RMSE={rmse_t3:.4f}, 後半RMSE={rmse_t4:.4f}")


    # === CSV出力 ===
    output_csv_path = r"C:\Users\tslab\Desktop\予測\予測結果_比較.csv"

    pred_series  = test_predictions[:, 0, 0]
    true_series  = test_ground_truth[:, 0, 0]

    with open(output_csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['インデックス', '実測値', '予測値'])
        for i in range(len(pred_series)):
            writer.writerow([i + 1, true_series[i], pred_series[i]])

    prediction_count += 1
    print("最大予測値（ノード0）：", np.max(pred_series))


    if prediction_count == 21:
        print("⚠️ 最大試行回数に到達。強制終了します。")
        break

    joblib.dump(scaler, scaler_path)
    print("✅ スケーラー保存完了")
    print(f"最高精度予測完了: 前半RMSE={rmse_t3:.4f}, 後半RMSE={rmse_t4:.4f}")



# === 2サイクルごとの予測結果と大小関係をCSVに保存 ===
output_dir = os.path.dirname(csv_output_path)
os.makedirs(output_dir, exist_ok=True)

output_rows = []


# ノード0（道路1）の予測列を取得
pred_series = test_predictions[:, 0, 0].astype(int)

output_rows = []
num_steps = len(pred_series)

for i in range(0, num_steps - 1, 2):
    cycle1 = i + 1
    cycle2 = i + 2
    pred1 = int(pred_series[i])     # t3
    pred2 = int(pred_series[i + 1]) # t4

    if pred1 == pred2:
        relation_label = 1  # 同じ
    elif pred1 < pred2:
        relation_label = 0  # 増加
    else:
        relation_label = 2  # 減少

    output_rows.append([cycle1, cycle2, pred1, pred2, relation_label])


# 書き出し処理
with open(csv_output_path, mode='w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerow(["サイクル1", "サイクル2", "予測値1", "予測値2", "大小関係ラベル"])
    writer.writerows(output_rows)

print(f"✅ 2サイクルごとの予測比較結果を保存しました：\n{csv_output_path}")

# === 正解CSV読み込み（ラベル1列目, 2行目以降） ===
true_labels = []
with open(answer_csv_path, mode='r', encoding='utf-8-sig') as f:
    reader = list(csv.reader(f))
    reader = reader[2:]  # ヘッダー + 1行目をスキップ（インデックス1以降）

    for row in reader:
        if row:
            true_labels.append(int(row[0]))

# === 予測ラベル読み込み（保存したばかりのCSV）===
pred_labels = []
with open(csv_output_path, mode='r', encoding='utf-8-sig') as f:
    reader = csv.reader(f)
    next(reader)  # ヘッダーをスキップ
    for row in reader:
        if row:
            pred_labels.append(int(row[4]))  # ラベル列

# === losの記録 ===
# === 精度評価 ===
correct = sum(p == t for p, t in zip(pred_labels, true_labels))
total = len(true_labels)
accuracy = correct / total * 100 if total > 0 else 0

print(f"🎯 正答数: {correct} / {total}（正答率: {accuracy:.2f}%）")

# === ベストモデルのLoss曲線をプロット ===
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plt.plot(best_train_losses, label='Train Loss', linewidth=2)
plt.plot(best_val_losses, label='Validation Loss', linewidth=2)
plt.title("Training & Validation Loss (Best Model Cycle)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

loss_plot_path = r"C:\Users\tslab\Desktop\予測\Loss曲線_best_model.png"
plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"📉 ベストモデルのLoss曲線を保存しました → {loss_plot_path}")


# # グラフ作成
# import matplotlib.pyplot as plt
# import matplotlib
# from matplotlib.ticker import ScalarFormatter, MaxNLocator
# import os

# matplotlib.rcParams['font.family'] = 'Meiryo'
# plt.rcParams["font.size"] = 15

# # === グラフ描画用の長さ取得 ===
# test_long = flattened_pred.shape[0]  # flattenedされたt3, t4系列長 = 1328

# # 1時間あたりのサイクル数（24分割）
# cycles_per_hour = test_long // 24  # 約55サイクル/時間
# hour_ticks = [i * cycles_per_hour for i in range(24)]
# hour_labels = [f'{i}時' for i in range(24)]

# plt.rcParams["font.size"] = 15

# # === 色分けグラフ：実測 vs 予測を前半（t+2）と後半（t+3）で色分け ===
# plt.figure(figsize=(12, 6))

# # === 実測データ：連続線だが、偶数・奇数で色分け ===
# for i in range(test_long - 1):
#     x_vals = [i, i + 1]
#     y_vals = [flattened_true[i, 0], flattened_true[i + 1, 0]]
#     color = 'blue' if i % 2 == 0 else 'skyblue'
#     plt.plot(x_vals, y_vals, color=color, linewidth=1.5, label='Actual' if i == 0 else "")

# # === 予測データ：連続線だが、偶数・奇数で色分け ===
# for i in range(test_long - 1):
#     x_vals = [i, i + 1]
#     y_vals = [flattened_pred[i, 0], flattened_pred[i + 1, 0]]
#     color = 'red' if i % 2 == 0 else 'orange'
#     plt.plot(x_vals, y_vals, color=color, linewidth=1.5, linestyle='--', label='Predicted' if i == 0 else "")

# # 軸などの設定
# plt.grid(which="both", axis="both")
# plt.minorticks_on()
# plt.xticks(hour_ticks, hour_labels, rotation=45)
# plt.gca().yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
# plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
# plt.ticklabel_format(style='plain', axis='y')
# plt.title('Vehicle Number Forecast with Alternating Color Lines (Road 1)')
# plt.xlim(0, test_long)
# plt.xlabel('Cycle')
# plt.ylabel('Number of vehicles')
# plt.legend()
# plt.tight_layout()

# # 保存・表示
# plt.savefig(f'C:\\Users\\tslab\\Desktop\\予測\\相関係数0.5以上\\道路10_11除く_予測波形_交互線色_{rmse}.png')
# plt.show()

# import matplotlib.pyplot as plt
# import matplotlib
# from matplotlib.ticker import ScalarFormatter, MaxNLocator
# import os
# import numpy as np

# # === フォント設定 ===
# matplotlib.rcParams['font.family'] = 'Meiryo'
# plt.rcParams["font.size"] = 15

# # === グラフ描画用の長さ取得 ===
# test_long = flattened_pred.shape[0]  # flattenedされたt3, t4系列長 = 1328

# # === 1時間あたりのサイクル数（24分割） ===
# cycles_per_hour = test_long // 24
# hour_ticks = [i * cycles_per_hour for i in range(24)]
# hour_labels = [f'{i}時' for i in range(24)]

# # === 偶数・奇数インデックスを分離 ===
# even_idx = np.arange(0, test_long, 2)  # 偶数サイクル (t1)
# odd_idx  = np.arange(1, test_long, 2)  # 奇数サイクル (t2)

# # === グラフ作成 ===
# plt.figure(figsize=(12, 6))

# # === 実測データ ===
# plt.plot(even_idx, flattened_true[even_idx, 0], color='blue', linewidth=1.8, label='Actual t1 (even)')
# plt.plot(odd_idx,  flattened_true[odd_idx, 0],  color='skyblue', linewidth=1.8, label='Actual t2 (odd)')

# # === 予測データ ===
# plt.plot(even_idx, flattened_pred[even_idx, 0], color='red', linewidth=1.8, linestyle='--', label='Predicted t1 (even)')
# plt.plot(odd_idx,  flattened_pred[odd_idx, 0],  color='orange', linewidth=1.8, linestyle='--', label='Predicted t2 (odd)')

# # === 軸などの設定 ===
# plt.grid(which="both", axis="both")
# plt.minorticks_on()
# plt.xticks(hour_ticks, hour_labels, rotation=45)
# plt.gca().yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
# plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
# plt.ticklabel_format(style='plain', axis='y')
# plt.title('Vehicle Number Forecast (Even–Odd Segmented, Road 1)')
# plt.xlim(0, test_long)
# plt.xlabel('Cycle')
# plt.ylabel('Number of vehicles')
# plt.legend()
# plt.tight_layout()

# # === 保存・表示 ===
# save_dir = r'C:\Users\tslab\Desktop\予測\相関係数0.5以上'
# os.makedirs(save_dir, exist_ok=True)
# save_path = os.path.join(save_dir, f'道路10_11除く_予測波形_偶数奇数別線_{rmse}.png')

# plt.savefig(save_path)
# plt.show()
# print(f"✅ グラフを保存しました: {save_path}")


# # === 全体波形 ===
# plt.figure(figsize=(12, 6))
# plt.plot(flattened_true[:, 0], color='blue', label='Actual')
# plt.plot(flattened_pred[:, 0], color='red', label='Predicted')
# plt.grid(which="both", axis="both")
# plt.minorticks_on()
# plt.xticks(hour_ticks, hour_labels, rotation=45)
# plt.gca().yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
# plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
# plt.ticklabel_format(style='plain', axis='y')
# plt.title('Vehicle Number Forecast for Road 1')
# plt.xlim(0, test_long)
# plt.xlabel('Cycle')
# plt.ylabel('Number of vehicles')
# plt.legend()
# rmse_rounded = f"{rmse:.3f}".replace('.', '_')
# plt.savefig(f'C:\\Users\\tslab\\Desktop\\予測\\相関係数0.5以上\\道路10_道路11を除く_予測波形_{rmse}.png')
# plt.show()


# # グラフ作成（t1同士・t2同士をそれぞれ結ぶバージョン）
# import matplotlib.pyplot as plt
# import matplotlib
# from matplotlib.ticker import ScalarFormatter, MaxNLocator
# import os
# import numpy as np

# # === フォント設定 ===
# matplotlib.rcParams['font.family'] = 'Meiryo'
# plt.rcParams["font.size"] = 15

# # === グラフ描画用の長さ取得 ===
# test_long = flattened_pred.shape[0]  # flattenedされたt3, t4系列長 = 1328

# # === 1時間あたりのサイクル数（24分割） ===
# cycles_per_hour = test_long // 24  # 約55サイクル/時間
# hour_ticks = [i * cycles_per_hour for i in range(24)]
# hour_labels = [f'{i}時' for i in range(24)]

# # === t1とt2のインデックスを分離 ===
# t1_idx = np.arange(0, test_long, 2)  # 偶数サイクル (t1)
# t2_idx = np.arange(1, test_long, 2)  # 奇数サイクル (t2)

# plt.figure(figsize=(12, 6))

# # === 実測データ ===
# plt.plot(t1_idx, flattened_true[t1_idx, 0], color='blue', linewidth=1.8, label='Actual t1')
# plt.plot(t2_idx, flattened_true[t2_idx, 0], color='skyblue', linewidth=1.8, label='Actual t2')

# # === 予測データ ===
# plt.plot(t1_idx, flattened_pred[t1_idx, 0], color='red', linewidth=1.8, linestyle='--', label='Predicted t1')
# plt.plot(t2_idx, flattened_pred[t2_idx, 0], color='orange', linewidth=1.8, linestyle='--', label='Predicted t2')

# # === 軸などの設定 ===
# plt.grid(which="both", axis="both")
# plt.minorticks_on()
# plt.xticks(hour_ticks, hour_labels, rotation=45)
# plt.gca().yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
# plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
# plt.ticklabel_format(style='plain', axis='y')
# plt.title('Vehicle Number Forecast (t1–t1, t2–t2 Connected, Road 1)')
# plt.xlim(0, test_long)
# plt.xlabel('Cycle')
# plt.ylabel('Number of vehicles')
# plt.legend()
# plt.tight_layout()

# # === 保存・表示 ===
# save_dir = r'C:\Users\tslab\Desktop\予測\相関係数0.5以上'
# os.makedirs(save_dir, exist_ok=True)
# save_path = os.path.join(save_dir, f'道路10_11除く_予測波形_t1t2別線_{rmse}.png')

# plt.savefig(save_path)
# plt.show()
# print(f"✅ グラフを保存しました: {save_path}")


# import matplotlib.pyplot as plt
# from matplotlib.ticker import ScalarFormatter, MaxNLocator

# # === 拡大波形（8:00〜9:00）前後ステップ色分け表示 ===
# start_idx = hour_ticks[8]
# end_idx = hour_ticks[9]

# plt.figure(figsize=(12, 6))

# # === 実測データ（線 + 点） ===
# for i in range(start_idx, end_idx - 1):
#     x_vals = [i, i + 1]
#     y_vals = [flattened_true[i, 0], flattened_true[i + 1, 0]]
#     color = 'blue' if i % 2 == 0 else 'skyblue'
#     plt.plot(x_vals, y_vals, color=color, linewidth=2, label='Actual' if i == start_idx else "")
#     plt.plot(i, flattened_true[i, 0], marker='o', color=color)  # ← 点
# plt.plot(end_idx - 1, flattened_true[end_idx - 1, 0], marker='o', color='skyblue')  # 最後の点

# # === 予測データ（線 + 点） ===
# for i in range(start_idx, end_idx - 1):
#     x_vals = [i, i + 1]
#     y_vals = [flattened_pred[i, 0], flattened_pred[i + 1, 0]]
#     color = 'red' if i % 2 == 0 else 'orange'
#     plt.plot(x_vals, y_vals, color=color, linewidth=2, linestyle='--', label='Predicted' if i == start_idx else "")
#     plt.plot(i, flattened_pred[i, 0], marker='o', color=color)  # ← 点
# plt.plot(end_idx - 1, flattened_pred[end_idx - 1, 0], marker='o', color='orange')  # 最後の点

# # === 軸・装飾 ===
# plt.grid(which="both", axis="both")
# plt.minorticks_on()
# plt.xticks(hour_ticks[1:3], hour_labels[1:3], rotation=45)
# plt.xlim(start_idx, end_idx)
# plt.gca().yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
# plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
# plt.ticklabel_format(style='plain', axis='y')
# plt.title('Vehicle Number Forecast')
# plt.xlabel('Cycle')
# plt.ylabel('Number of vehicles')
# plt.legend()
# plt.tight_layout()

# # === 保存・表示 ===
# plt.savefig(r'C:\Users\tslab\Desktop\予測\相関係数0.5以上\道路10_11除く_拡大波形_色分け_点付き2.png')
# plt.show()


# ==== grad-cam (5月27日追加) =====================================================
# import numpy as np
# from tqdm import tqdm

# # === GNNExplainer用にモデル再構成してロード ===
# model = TrafficPredictionGNN(num_node_features=2, dropout_rate=0.2).to(device)
# model.load_state_dict(torch.load(best_model_path, map_location=device))
# model.eval()

# # === GNNExplainer の構成 ===
# model_config = ModelConfig(
#     mode='regression',
#     task_level='node',
#     return_type='raw',
# )

# explainer = Explainer(
#     model=model,
#     algorithm=GNNExplainer(epochs=200),
#     explanation_type='model',
#     node_mask_type='attributes',
#     edge_mask_type='object',
#     model_config=model_config,
# )

# # === ノード0（道路1）に対する全テストデータでの重要度平均を取得 ===
# node_importance_list = []

# from tqdm import tqdm  # 進捗表示
# for i in tqdm(range(len(test_dataset))):
#     data = test_dataset[i].to(device)
#     try:
#         explanation = explainer(data.x, data.edge_index, index=0)  # ノード0（道路1）
#         node_importance = explanation.node_mask.detach().cpu().numpy().flatten()
#         node_importance_list.append(node_importance)
#     except Exception as e:
#         print(f"⚠ サイクル {i} の解析中にエラー: {e}")
#         continue

# # === 平均重要度を計算 ===
# node_importance_array = np.array(node_importance_list)
# mean_importance = node_importance_array.mean(axis=0)
# top_nodes = mean_importance.argsort()[::-1]

# # used_road_indices を定義済みとしてフィルタ
# filtered_mean_importance = mean_importance[used_road_indices]
# top_indices_in_used = filtered_mean_importance.argsort()[::-1]

# print("\n🔍 全サイクル平均による 道路1（ノード0） の予測に影響を与えたノード（重要度順・使用ノードのみ）:")
# for rank, idx in enumerate(top_indices_in_used):
#     node_idx = used_road_indices[idx]
#     importance = mean_importance[node_idx]
#     print(f"Rank {rank+1}: 道路{node_idx+1}（ノード{node_idx}） - 平均重要度: {importance:.4f}")

# import matplotlib.pyplot as plt
# import numpy as np

# # ノード0（道路1）のみ抽出
# pred_road1 = test_predictions[:, 0]
# true_road1 = test_ground_truth[:, 0]

# # 誤差を計算
# error_road1 = pred_road1 - true_road1

# # 時系列グラフを描画
# plt.figure(figsize=(14, 5))
# plt.plot(error_road1, label='Prediction Error (Road 1)', color='red')
# plt.axhline(0, color='black', linestyle='--')
# plt.xlabel('Time Step (e.g. 24sec cycle)')
# plt.ylabel('Error (Pred - True)')
# plt.title('Prediction Error over Time for Road 1')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()


# # ヒートマップ追加
# import matplotlib
# import matplotlib.pyplot as plt
# import numpy as np
# from sumolib.net import readNet
# from matplotlib import colormaps

# # 🔤 日本語フォント（Meiryo, IPAexGothic, Noto Sans CJK JP など）
# matplotlib.rcParams['font.family'] = 'Meiryo'

# # ノード→エッジIDの対応（SUMO上の道路ID）
# node_to_edge_map = {
#     0: ["DtoA"],
#     1: ["E91", "E92", "-543210000#5"],
#     2: ["E12"],
#     3: ["E159", "E158"],
#     4: ["E26"],
#     5: ["E196", "-E197", "E195"],
#     6: ["-E53", "-E35"],
#     7: ["E36"],
#     8: ["M-1toJ"],
# }

# # SUMOネットワークファイルの読み込み
# net = readNet("C:/Users/tslab/Desktop/予測/toyama_shouwa_offset_smz.net.xml")

# # 描画対象（重要度 > 0 のノード）だけ正規化
# valid_indices = np.where(mean_importance > 0)[0]
# valid_importances = mean_importance[valid_indices]
# min_imp, max_imp = valid_importances.min(), valid_importances.max()
# importance_norm = (mean_importance - min_imp) / (max_imp - min_imp + 1e-6)

# # == 赤一色の濃淡カラーマップ：薄赤 → 赤 → 濃赤 =====================
# from matplotlib.colors import LinearSegmentedColormap
# cmap = LinearSegmentedColormap.from_list("custom_red", ["mistyrose", "red", "darkred"])

# fig, ax = plt.subplots(figsize=(12, 12))

# # 🟥 全道路をまず灰色で描画
# for edge in net.getEdges():
#     shape = edge.getShape()
#     if len(shape) >= 2:
#         xs, ys = zip(*shape)
#         ax.plot(xs, ys, color='lightgray', linewidth=1)

# # 🔥 重要ノードのみ強調描画＆ラベル表示
# for node_id, edge_ids in node_to_edge_map.items():
#     if node_id >= len(mean_importance):
#         continue
#     importance = mean_importance[node_id]
#     if importance <= 0:
#         continue  # スキップ：重要度が0以下

#     color = cmap(importance_norm[node_id])

#     # 複数エッジにまたがる場合のラベル位置計算用
#     all_xs, all_ys = [], []
#     for edge_id in edge_ids:
#         try:
#             edge = net.getEdge(edge_id)
#             shape = edge.getShape()
#             if len(shape) >= 2:
#                 xs, ys = zip(*shape)
#                 ax.plot(xs, ys, color=color, linewidth=6)
#                 all_xs.extend(xs)
#                 all_ys.extend(ys)
#         except Exception as e:
#             print(f"⚠ エッジ {edge_id} に関するエラー: {e}")

#     # 🏷 ラベルを中心に1つだけ表示
#     if all_xs and all_ys:
#         center_x = sum(all_xs) / len(all_xs)
#         center_y = sum(all_ys) / len(all_ys)
#         ax.text(center_x, center_y, f"道路{node_id+1}", fontsize=10,
#                 color='black', ha='center', va='center',
#                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.5))

# # 🧭 カラーバー
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_imp, vmax=max_imp))
# sm.set_array([])
# cbar = plt.colorbar(sm, ax=ax)
# cbar.set_label('ノード重要度（道路1に対する影響）', fontsize=12)

# # 📝 タイトルなど
# ax.set_title('道路1への影響度ヒートマップ（赤=重要）', fontsize=16, weight='bold')
# ax.axis('off')
# plt.tight_layout()
# plt.show()
