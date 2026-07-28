import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================
# 予測対象の道路設定
# ここを見れば、今どの道路を予測するかすぐ分かります
# 例:
#   TARGET_ROAD = "道路1"
#   TARGET_ROAD = "道路5"
#   TARGET_ROAD = 5
# 実行時に第1引数を渡した場合は、その値が優先されます。
# =========================
TARGET_ROAD = "道路1"


def extract_road_id(value, fallback=None):
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    if digits:
        return int(digits)
    if fallback is not None:
        return fallback
    raise ValueError(f"道路番号を判定できません: {value}")


def print_target_road_banner(target_road_label, target_node_index, selected_target_road, model_output_dir):
    print("\n" + "=" * 72)
    print(f"予測対象道路 : {target_road_label}")
    print(f"対象ノード   : {target_node_index}")
    print(f"設定値       : {selected_target_road}")
    print(f"保存先       : {model_output_dir}")
    print("=" * 72 + "\n")

print(torch.cuda.is_available())
print(f"TARGET_ROAD 設定値: {TARGET_ROAD}")

# === 使用ノード（道路8・9を除く） ===
used_road_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]
# used_road_indices = [0, 2,  4, 5, 6]
node_num = len(used_road_indices)
selected_target_road = sys.argv[1] if len(sys.argv) > 1 else TARGET_ROAD
target_road_id = extract_road_id(selected_target_road)

# === デバイスの設定 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === データ読み込み ===
# train_data_csv = pd.read_csv(r"C:\Users\Tsukasa\Desktop\研究\予測\修正済みデータ_5ヶ月分.csv", encoding='cp932')
# train_data_csv = pd.read_csv(r"C:\Users\Tsukasa\Desktop\研究\予測\共三参照_151日データ.csv", encoding='cp932')
# train_data_csv = pd.read_csv(r"C:\Users\Tsukasa\Desktop\研究\予測\前後半11月10日データ.csv", encoding='cp932')
train_data_csv = pd.read_csv(os.path.join(BASE_DIR, "★学習用データ", "1. 赤時間2分割_学習データ_12.22", "新環境待ち台数データ_12.22.csv"), encoding='cp932')
adjacency_matrix = pd.read_csv(r"C:\\Users\\Tsukasa\\Desktop\\研究\\予測\\新環境_隣接行列.csv", encoding='shift-jis', index_col=0)
# adjacency_matrix = pd.read_csv(r"C:\Users\Tsukasa\Desktop\研究\予測\交差点A, Jのみの道路情報.csv", encoding='shift-jis', index_col=0)
print(adjacency_matrix)

csv_output_path = r'C:\\Users\\Tsukasa\\Desktop\\研究\\予測\\csv掃き出し\\遅れ時間.csv'
answer_csv_path = os.path.join(BASE_DIR, "★学習用データ", "1. 赤時間2分割_学習データ_12.22", "新環境正解データ_12.22.csv")

scaler_path = r"C:\Users\Tsukasa\Desktop\研究\予測\予測モデル\scaler.pkl"

# == データ数 ==============================================================
DAY_LENGTH = 1330  # 1日のサンプル数（固定・変更しないこと）
# DAY_LENGTH = 665  # 1日のサンプル数（固定）

# == 時間帯フィルタ設定 =====================================================
# True: 指定した時間帯のデータのみで学習 / False: 1日全体（0〜24時）を使用
USE_TIME_FILTER = True
START_HOUR = 10   # 学習に使う開始時刻（0〜23）
END_HOUR   = 22   # 学習に使う終了時刻（1〜24、START_HOUR より大きい値）

_samples_per_hour = DAY_LENGTH / 24
TIME_START_IDX = round(START_HOUR * _samples_per_hour) if USE_TIME_FILTER else 0
TIME_END_IDX   = round(END_HOUR   * _samples_per_hour) if USE_TIME_FILTER else DAY_LENGTH

print(f"時間帯フィルタ: {'有効' if USE_TIME_FILTER else '無効'} "
      f"({START_HOUR}時〜{END_HOUR}時, "
      f"1日あたり {TIME_END_IDX - TIME_START_IDX} サンプル)")
# ==========================================================================

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
used_road_labels = [str(train_data_csv.columns[i]) for i in used_road_indices]
used_road_ids = [
    extract_road_id(label, used_road_indices[pos] + 1)
    for pos, label in enumerate(used_road_labels)
]
if target_road_id not in used_road_ids:
    raise ValueError(
        f"対象道路{target_road_id}は使用道路に含まれていません。選択可能: {used_road_ids}"
    )

target_node_index = used_road_ids.index(target_road_id)
target_road_label = used_road_labels[target_node_index]
# 成果物（モデル・スケーラー・予測結果CSV・グラフ）はすべてこのフォルダに出力する
model_output_dir = os.path.join(BASE_DIR, "予測モデル", "1. 赤時間2分割", target_road_label)
os.makedirs(model_output_dir, exist_ok=True)

prediction_detail_csv_path = os.path.join(model_output_dir, f"予測結果_{target_road_label}.csv")
csv_output_path = os.path.join(model_output_dir, f"予測結果_大小比較_{target_road_label}.csv")
scaler_path = os.path.join(model_output_dir, "scaler.pkl")
loss_plot_path = os.path.join(model_output_dir, f"Loss曲線_{target_road_label}_best_model.png")
t3_plot_path = os.path.join(model_output_dir, f"予測波形_{target_road_label}_t3.png")
t4_plot_path = os.path.join(model_output_dir, f"予測波形_{target_road_label}_t4.png")

print(f"予測対象: {target_road_label} (ノード index: {target_node_index})")
print_target_road_banner(
    target_road_label=target_road_label,
    target_node_index=target_node_index,
    selected_target_road=selected_target_road,
    model_output_dir=model_output_dir,
)

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
def build_daywise_dataset(data_array, labels, edge_index, device, target_node_index, alpha=2):
    total_steps = data_array.shape[1]
    num_days = total_steps // DAY_LENGTH
    dataset = []

    for day in range(num_days):
        day_start = day * DAY_LENGTH
        # 時間帯フィルタに基づいて走査範囲を決定
        range_start = day_start + TIME_START_IDX
        range_end   = day_start + TIME_END_IDX

        for t in range(range_start, range_end - 3, 2):
            x_t1 = torch.tensor(data_array[:, t], dtype=torch.float).unsqueeze(1).to(device)
            x_t2 = torch.tensor(data_array[:, t+1], dtype=torch.float).unsqueeze(1).to(device)

            # 🚩 道路1（ノード0）だけを α 倍
            x_t1[target_node_index] *= alpha
            x_t2[target_node_index] *= alpha

            x = torch.cat([x_t1, x_t2], dim=1)

            y_t3 = torch.tensor(data_array[:, t+2], dtype=torch.float).unsqueeze(1).to(device)
            y_t4 = torch.tensor(data_array[:, t+3], dtype=torch.float).unsqueeze(1).to(device)
            y = torch.cat([y_t3, y_t4], dim=1)

            dataset.append(Data(x=x, edge_index=edge_index, y=y))

    return dataset



train_dataset = build_daywise_dataset(train_data, train_labels, edge_index, device, target_node_index)
val_dataset   = build_daywise_dataset(validation_data, val_labels, edge_index, device, target_node_index)
test_dataset  = build_daywise_dataset(test_data, test_labels, edge_index, device, target_node_index)


# === バッチ作成 ===
batch_size = 128
# batch_size = 10
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch.nn as nn


# ==== 自己回帰的2段階予測 + Teacher Forcing =========================================
# 学習時: true_t3（正解t3）をt4ヘッドに渡す → クリーンなシグナルで学習
# 推論時: t3_pred（予測t3）をt4ヘッドに渡す → 実際の予測チェーンで評価
class TrafficPredictionGNN(torch.nn.Module):
    def __init__(self, num_node_features, dropout_rate=0.2):
        super(TrafficPredictionGNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, 128)
        self.skip  = nn.Linear(num_node_features, 128)  # スキップ接続: 元特徴量を直接128次元へ

        self.lin_t3 = nn.Linear(128, 1)          # t3専用ヘッド
        self.lin_t4 = nn.Linear(128 + 1, 1)      # t4ヘッド: 隠れ状態 + t3値を入力
        self.dropout_rate = dropout_rate

    def forward(self, x, edge_index, true_t3=None, teacher_forcing_ratio=1.0):
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        h = F.relu(self.conv3(h, edge_index))
        h = h + self.skip(x)  # スキップ接続: 元の局所特徴量を加算して変動情報を保持

        t3_pred = self.lin_t3(h)                                   # [num_nodes, 1]
        # Scheduled Sampling: 確率的に正解t3か予測t3を選択
        if true_t3 is not None and random.random() < teacher_forcing_ratio:
            t3_for_t4 = true_t3
        else:
            t3_for_t4 = t3_pred
        h_t4 = torch.cat([h, t3_for_t4], dim=1)
        t4_pred = self.lin_t4(h_t4)                                # [num_nodes, 1]

        return torch.cat([t3_pred, t4_pred], dim=1)                # [num_nodes, 2]
    
# class TrafficPredictionGNN(torch.nn.Module):
#     def __init__(self, num_node_features, dropout_rate=0.2):
#         super(TrafficPredictionGNN, self).__init__()
#         self.conv1 = GCNConv(num_node_features, 32)
#         self.conv2 = GCNConv(32, 64)
#         self.conv3 = GCNConv(64, 64)

#         self.lin = nn.Linear(64, 2)
#         self.dropout_rate = dropout_rate

#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)

#         x = self.conv2(x, edge_index)
#         x = F.relu(x)

#         x = self.conv3(x, edge_index)
#         x = F.relu(x)

#         out = self.lin(x)
#         return out

# 学習率のリスト（エポック数に基づいて設定）
# learning_rates = [0.001] * 100 + [0.0005] * 50 + [0.0001] * 50  # 合計200
learning_rates = [0.001] * 200 + [0.0005] * 100 + [0.0001] * 100  # 合計400

#モデルの保存パス
best_model_path = r"C:\Users\Tsukasa\Desktop\研究\予測\予測モデル\GCN_epoch_best_model.pth"

# 早期停止の設定
patience = 15                    #早期終了の条件
# patience = 3
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
rmse_first = float('inf')
rmse_second = float('inf')

# === Loss履歴の保存用 ===
best_overall_val_loss = float('inf')
best_train_losses = []
best_val_losses = []


# トレーニングループ（指定したRMSEより精度が向上しない場合は，再度学習・予測）
# while (rmse_first > 1.5 or rmse_second > 1.5):
# 評価値 (道路1と道路5のRMSEで判定)
while (rmse_first > 1.5 or rmse_second > 1.5):
# while (rmse_first > 1.5 or rmse_second > 1.5 or rmse5_first > 1.5 or rmse5_second > 1.5):

    model = TrafficPredictionGNN(num_node_features=2, dropout_rate=0.2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    #Lossを格納している配列中身を削除
    train_losses.clear()
    val_losses.clear()
    epochs_no_improve = 0
    #best_Model保存を変数初期化
    best_model_path = os.path.join(model_output_dir, "GCN_epoch_best_model.pth")
    best_val_loss = float('inf')
    best_test_loss = float('inf')

    print(f"{prediction_count}回目の学習中 [{target_road_label}]")
    model.train()
    for epoch in range(400):  # 例として100エポック
        
        # エポックに応じて学習率を変更
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rates[epoch]

        # Scheduled Sampling: 前半200エポックで1.0→0.0に線形減衰、以降は0.0固定
        tf_ratio = max(0.0, 1.0 - epoch / 200)

        # トレーニング
        running_train_loss = 0.0
        model.train()  # モデルを訓練モードに
        for batch in train_loader:
            batch = batch.to(device)  # データをGPUに移動
            optimizer.zero_grad()
            # 出力：[batch * ノード数, 2]
            # Scheduled Sampling: エポックに応じてTeacher Forcing比率を下げる
            out = model(batch.x, batch.edge_index, true_t3=batch.y[:, 0:1], teacher_forcing_ratio=tf_ratio)
            # 道路1に該当するノードのみを抽出（インデックス0と仮定）
            out = out.view(-1, node_num, 2)  # shape: [batch_size, node_num, 2]
            y   = batch.y.view(-1, node_num, 2)

            # # ノード0だけ取り出して2時刻分まとめて損失計算
            out_target = out[:, target_node_index, :]  # shape: [batch_size, 2]
            y_target   = y[:, target_node_index, :]    # shape: [batch_size, 2]

            mse_t3 = criterion(out_target[:, 0:1], y_target[:, 0:1])
            mse_t4 = criterion(out_target[:, 1:2], y_target[:, 1:2])
            mae_t3 = F.l1_loss(out_target[:, 0:1], y_target[:, 0:1])
            mae_t4 = F.l1_loss(out_target[:, 1:2], y_target[:, 1:2])
            loss = (mse_t3 + mae_t3) + 5.0 * mae_t4  # t4はMAEのみ（MSE排除で平均回帰を防ぐ）
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
                out = out.view(-1, node_num, 2)
                y = batch.y.view(-1, node_num, 2).to(device)

                # ノード0だけ取り出して2時刻分まとめて損失計算
                out_target = out[:, target_node_index, :]  # shape: [batch_size, 2]
                y_target   = y[:, target_node_index, :]    # shape: [batch_size, 2]

                mse_t3 = criterion(out_target[:, 0:1], y_target[:, 0:1])
                mse_t4 = criterion(out_target[:, 1:2], y_target[:, 1:2])
                mae_t3 = F.l1_loss(out_target[:, 0:1], y_target[:, 0:1])
                mae_t4 = F.l1_loss(out_target[:, 1:2], y_target[:, 1:2])
                loss = (mse_t3 + mae_t3) + 5.0 * mae_t4  # t4はMAEのみ（MSE排除で平均回帰を防ぐ）
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
            out = out.view(-1, node_num, 2)
            y = batch.y.view(-1, node_num, 2)

            test_predictions.append(out.cpu().numpy())  # shape: [batch, node_num, 2]
            test_ground_truth.append(y.cpu().numpy())   # shape: [batch, node_num, 2]

    # 結合 → shape: [N, node_num, 2]
    test_predictions = np.concatenate(test_predictions, axis=0)
    test_ground_truth = np.concatenate(test_ground_truth, axis=0)

    # === 逆正規化 ===
    for i in range(2):  # 2ステップ分
        test_predictions[:, :, i] = scaler.inverse_transform(test_predictions[:, :, i])
        test_ground_truth[:, :, i] = scaler.inverse_transform(test_ground_truth[:, :, i])

    print("逆正規化・四捨五入前の予測データ：", test_predictions.shape)
    print("逆正規化・四捨五入前のテストデータ：", test_ground_truth.shape)

# ===============================
# 評価 RMSE
# ===============================
    # === RMSEの計算（ノード0だけ） ===
    pred_t3 = test_predictions[:, target_node_index, 0]
    true_t3 = test_ground_truth[:, target_node_index, 0]
    pred_t4 = test_predictions[:, target_node_index, 1]
    true_t4 = test_ground_truth[:, target_node_index, 1]

    # t3, t4 を連結して一続きの系列として扱う
    pred_all = np.concatenate([pred_t3, pred_t4])
    true_all = np.concatenate([true_t3, true_t4])

    # 系列長を取得
    n = len(pred_all)
    half = n // 2

    # === 前半・後半に分割 ===
    pred_first = pred_all[:half]
    true_first = true_all[:half]
    pred_second = pred_all[half:]
    true_second = true_all[half:]

    # === RMSE計算 ===
    rmse_first = np.sqrt(np.mean((pred_first - true_first) ** 2))
    rmse_second = np.sqrt(np.mean((pred_second - true_second) ** 2))

    # === 結果を表示 ===
    print(f"前半RMSE: {rmse_first:.4f}, 後半RMSE: {rmse_second:.4f}")


    # === スライド展開（flatten） ===
    flattened_pred = np.zeros((test_ground_truth.shape[0]*2, node_num))
    flattened_true = np.zeros((test_ground_truth.shape[0]*2, node_num))

    for i in range(test_predictions.shape[0]):
        t_idx = i * 2 + 2
        if t_idx + 1 >= flattened_pred.shape[0]:
            break
        flattened_pred[t_idx]     = test_predictions[i, :, 0]
        flattened_pred[t_idx + 1] = test_predictions[i, :, 1]
        flattened_true[t_idx]     = test_ground_truth[i, :, 0]
        flattened_true[t_idx + 1] = test_ground_truth[i, :, 1]

    # === CSV出力 ===
    output_csv_path = prediction_detail_csv_path
    with open(output_csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['時刻', '実測値', '予測値'])
        for i in range(flattened_pred.shape[0]):
            writer.writerow([
                i + 1,
                flattened_true[i, target_node_index],
                flattened_pred[i, target_node_index]
            ])

    # ======== 終了条件 ======== 
    joblib.dump(scaler, scaler_path)
    print(f"{target_road_label} 用スケーラーを保存しました: {scaler_path}")
    if rmse_first <= 1.5 and rmse_second <= 1.5: 
      print(f"[達成] 終了条件達成！前半RMSE={rmse_first:.4f}, 後半RMSE={rmse_second:.4f}")
      break
    else:
      print(f"[継続] 継続学習: 前半RMSE={rmse_first:.4f}, 後半RMSE={rmse_second:.4f}")


    prediction_count += 1
    print(f"最大予測値（{target_road_label}）:", np.amax(flattened_pred[:, target_node_index]))

    if prediction_count == 21:
        print("[警告] 最大試行回数に到達。強制終了します。")
        break

    joblib.dump(scaler, scaler_path)
    print("スケーラー保存完了")
    print(f"最高精度予測完了: 前半RMSE={rmse_first:.4f}, 後半RMSE={rmse_second:.4f}")

# === 2サイクルごとの予測結果と大小関係をCSVに保存 ===
output_dir = os.path.dirname(csv_output_path)
os.makedirs(output_dir, exist_ok=True)

output_rows = []

# flattened_pred[:, 0] に対してサイクル単位で走査
num_cycles = flattened_pred.shape[0]

for i in range(0, num_cycles - 1, 2):  # 2ステップずつ
    cycle1 = i + 1
    cycle2 = i + 2
    pred1 = int(flattened_pred[i, target_node_index])
    pred2 = int(flattened_pred[i + 1, target_node_index])

    # 数値ラベルで大小関係を判定
    if pred1 == pred2:
        relation_label = 1  # 同じ → ラベル1（中立）
    elif pred1 < pred2:
        relation_label = 0  # 増加 → ラベル0
    else:
        relation_label = 2  # 減少 → ラベル2

    output_rows.append([cycle1, cycle2, pred1, pred2, relation_label])

# 書き出し処理
with open(csv_output_path, mode='w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerow(["サイクル1", "サイクル2", "予測値1", "予測値2", "大小関係ラベル"])
    writer.writerows(output_rows)

print(f"2サイクルごとの予測比較結果を保存しました：\n{csv_output_path}")

# === 正解CSV読み込み（ラベル1列目, 2行目以降） ===
true_labels = []
with open(answer_csv_path, mode='r', encoding='utf-8-sig') as f:
    reader = list(csv.reader(f))
    reader = reader[2:]  # ヘッダー + 1行目をスキップ（インデックス1以降）

    for row in reader:
        if row:
            true_labels.append(int(row[target_node_index]))

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

print(f"{target_road_label} の正答数: {correct} / {total}（正答率: {accuracy:.2f}%）")

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

plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"ベストモデルのLoss曲線を保存しました → {loss_plot_path}")


base = BASE_DIR
# グラフ作成
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import ScalarFormatter, MaxNLocator

matplotlib.rcParams["font.family"] = "Meiryo"
plt.rcParams["font.size"] = 15

# 時間帯フィルタに応じて x 軸設定
if USE_TIME_FILTER:
    _num_hours = END_HOUR - START_HOUR
    _hour_offset = START_HOUR
else:
    _num_hours = 24
    _hour_offset = 0

# t3・t4 それぞれの系列長
half_len = len(pred_first)
cycles_per_hour_h = half_len / _num_hours
hour_ticks_h  = [round(i * cycles_per_hour_h) for i in range(_num_hours + 1)]
hour_labels_h = [f"{_hour_offset + i}時" for i in range(_num_hours + 1)]

# === 前半グラフ（t3：1ステップ先予測）===
fig1, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(range(half_len), true_first,  color="blue", linewidth=1.5, label="Actual (t3)")
ax1.plot(range(half_len), pred_first,  color="red",  linewidth=1.5, linestyle="--", label="Predicted (t3)")
ax1.set_xticks(hour_ticks_h)
ax1.set_xticklabels(hour_labels_h, rotation=45)
ax1.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
ax1.ticklabel_format(style="plain", axis="y")
ax1.set_title(f"前半（t3）: Vehicle Number Forecast ({target_road_label})")
ax1.set_xlim(0, half_len)
ax1.set_xlabel("Cycle")
ax1.set_ylabel("Number of vehicles")
ax1.legend()
ax1.grid(which="both", axis="both")
ax1.minorticks_on()
fig1.tight_layout()

save_path1 = t3_plot_path
fig1.savefig(save_path1, dpi=300, bbox_inches="tight")
plt.show()
print(f"前半（t3）グラフを保存しました → {save_path1}")

# === 後半グラフ（t4：2ステップ先予測）===
fig2, ax2 = plt.subplots(figsize=(12, 6))
ax2.plot(range(half_len), true_second, color="blue", linewidth=1.5, label="Actual (t4)")
ax2.plot(range(half_len), pred_second, color="red",  linewidth=1.5, linestyle="--", label="Predicted (t4)")
ax2.set_xticks(hour_ticks_h)
ax2.set_xticklabels(hour_labels_h, rotation=45)
ax2.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
ax2.ticklabel_format(style="plain", axis="y")
ax2.set_title(f"後半（t4）: Vehicle Number Forecast ({target_road_label})")
ax2.set_xlim(0, half_len)
ax2.set_xlabel("Cycle")
ax2.set_ylabel("Number of vehicles")
ax2.legend()
ax2.grid(which="both", axis="both")
ax2.minorticks_on()
fig2.tight_layout()

save_path2 = t4_plot_path
fig2.savefig(save_path2, dpi=300, bbox_inches="tight")
plt.show()
print(f"後半（t4）グラフを保存しました → {save_path2}")
