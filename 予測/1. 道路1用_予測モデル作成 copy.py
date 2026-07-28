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
loss_plot_path_t3 = os.path.join(model_output_dir, f"Loss曲線_{target_road_label}_t3モデル.png")
loss_plot_path_t4 = os.path.join(model_output_dir, f"Loss曲線_{target_road_label}_t4モデル.png")
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

# === Dataset 作成（単入力・単出力）===
# input_offset: 入力時刻のオフセット（t3用=0→t1, t4用=1→t2）
# target_offset: 出力時刻のオフセット（t3用=2→t3, t4用=3→t4）
def build_single_input_dataset(data_array, edge_index, device, target_node_index,
                                input_offset, target_offset, alpha=2):
    total_steps = data_array.shape[1]
    num_days = total_steps // DAY_LENGTH
    dataset = []

    for day in range(num_days):
        day_start = day * DAY_LENGTH
        range_start = day_start + TIME_START_IDX
        range_end   = day_start + TIME_END_IDX

        for t in range(range_start, range_end - 3, 2):
            in_idx  = t + input_offset
            out_idx = t + target_offset
            if out_idx >= data_array.shape[1]:
                continue

            x = torch.tensor(data_array[:, in_idx], dtype=torch.float).unsqueeze(1).to(device)
            x[target_node_index] *= alpha

            y = torch.tensor(data_array[:, out_idx], dtype=torch.float).unsqueeze(1).to(device)

            dataset.append(Data(x=x, edge_index=edge_index, y=y))

    return dataset


# === バッチ作成 ===
batch_size = 64

# t3モデル用（t1 → t3）
train_loader_t3 = DataLoader(build_single_input_dataset(train_data,      edge_index, device, target_node_index, input_offset=0, target_offset=2), batch_size=batch_size, shuffle=False)
val_loader_t3   = DataLoader(build_single_input_dataset(validation_data, edge_index, device, target_node_index, input_offset=0, target_offset=2), batch_size=batch_size, shuffle=False)
test_loader_t3  = DataLoader(build_single_input_dataset(test_data,       edge_index, device, target_node_index, input_offset=0, target_offset=2), batch_size=batch_size, shuffle=False)

# t4モデル用（t2 → t4）
train_loader_t4 = DataLoader(build_single_input_dataset(train_data,      edge_index, device, target_node_index, input_offset=1, target_offset=3), batch_size=batch_size, shuffle=False)
val_loader_t4   = DataLoader(build_single_input_dataset(validation_data, edge_index, device, target_node_index, input_offset=1, target_offset=3), batch_size=batch_size, shuffle=False)
test_loader_t4  = DataLoader(build_single_input_dataset(test_data,       edge_index, device, target_node_index, input_offset=1, target_offset=3), batch_size=batch_size, shuffle=False)


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

        self.lin = nn.Linear(128, 1)
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

# 学習率のリスト
learning_rates = [0.001] * 100 + [0.0005] * 50 + [0.0001] * 50  # 合計400

# 早期停止
patience = 20


def run_training(train_loader, val_loader, model_save_path, label):
    model = TrafficPredictionGNN(num_node_features=1, dropout_rate=0.2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    best_val_loss = float('inf')
    best_test_loss = float('inf')
    epochs_no_improve = 0
    _train_losses, _val_losses = [], []
    _best_train, _best_val = [], []

    for epoch in range(200):
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rates[epoch]

        model.train()
        running_train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index).view(-1, node_num, 1)
            y   = batch.y.view(-1, node_num, 1)
            loss = criterion(out[:, target_node_index, 0], y[:, target_node_index, 0])
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)
        _train_losses.append(avg_train_loss)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index).view(-1, node_num, 1)
                y   = batch.y.view(-1, node_num, 1).to(device)
                loss = criterion(out[:, target_node_index, 0], y[:, target_node_index, 0])
                running_val_loss += loss.item()

        avg_val_loss = running_val_loss / len(val_loader)
        _val_losses.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            _best_train = _train_losses.copy()
            _best_val   = _val_losses.copy()
            print(f"[{label}] Best saved at epoch {epoch+1}, Val Loss: {best_val_loss:.6f}")

        print(f"[{label}] Epoch {epoch+1}, Train: {avg_train_loss:.6f}, Val: {avg_val_loss:.6f}, lr: {learning_rates[epoch]}")

        if avg_val_loss < best_test_loss:
            best_test_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f'[{label}] Early stopping')
            break

    model.load_state_dict(torch.load(model_save_path, map_location=device))
    return model, _best_train, _best_val


def predict_single(model, test_loader):
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index).view(-1, node_num, 1)
            y   = batch.y.view(-1, node_num, 1)
            preds.append(out.cpu().numpy())
            truths.append(y.cpu().numpy())
    pred_arr  = np.concatenate(preds,  axis=0)  # [N, node_num, 1]
    truth_arr = np.concatenate(truths, axis=0)
    pred_arr[:, :, 0]  = scaler.inverse_transform(pred_arr[:, :, 0])
    truth_arr[:, :, 0] = scaler.inverse_transform(truth_arr[:, :, 0])
    return pred_arr[:, target_node_index, 0], truth_arr[:, target_node_index, 0]


# === t3モデル学習（t1 → t3）===
best_model_path_t3 = os.path.join(model_output_dir, "GCN_t3_best_model.pth")
rmse_t3 = float('inf')
count_t3 = 1

while rmse_t3 > 1.2:
    print(f"{count_t3}回目のt3モデル学習 [{target_road_label}]")
    model_t3, best_train_t3, best_val_t3 = run_training(
        train_loader_t3, val_loader_t3, best_model_path_t3, "t3"
    )
    pred_t3, true_t3 = predict_single(model_t3, test_loader_t3)
    rmse_t3 = float(np.sqrt(np.mean((pred_t3 - true_t3) ** 2)))
    print(f"t3 RMSE: {rmse_t3:.4f}")
    if count_t3 >= 20:
        print("[警告] t3モデル最大試行回数に到達")
        break
    count_t3 += 1

plt.figure(figsize=(8, 5))
plt.plot(best_train_t3, label='Train Loss', linewidth=2)
plt.plot(best_val_t3,   label='Validation Loss', linewidth=2)
plt.title(f"t3モデル Loss ({target_road_label})")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig(loss_plot_path_t3, dpi=300, bbox_inches='tight')
plt.close()
print(f"t3 Loss曲線を保存しました → {loss_plot_path_t3}")


# === t4モデル学習（t2 → t4）===
best_model_path_t4 = os.path.join(model_output_dir, "GCN_t4_best_model.pth")
rmse_t4 = float('inf')
count_t4 = 1

while rmse_t4 > 1.3:
    print(f"{count_t4}回目のt4モデル学習 [{target_road_label}]")
    model_t4, best_train_t4, best_val_t4 = run_training(
        train_loader_t4, val_loader_t4, best_model_path_t4, "t4"
    )
    pred_t4, true_t4 = predict_single(model_t4, test_loader_t4)
    rmse_t4 = float(np.sqrt(np.mean((pred_t4 - true_t4) ** 2)))
    print(f"t4 RMSE: {rmse_t4:.4f}")
    if count_t4 >= 20:
        print("[警告] t4モデル最大試行回数に到達")
        break
    count_t4 += 1

plt.figure(figsize=(8, 5))
plt.plot(best_train_t4, label='Train Loss', linewidth=2)
plt.plot(best_val_t4,   label='Validation Loss', linewidth=2)
plt.title(f"t4モデル Loss ({target_road_label})")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig(loss_plot_path_t4, dpi=300, bbox_inches='tight')
plt.close()
print(f"t4 Loss曲線を保存しました → {loss_plot_path_t4}")

print(f"\n最終結果: t3 RMSE={rmse_t3:.4f}, t4 RMSE={rmse_t4:.4f}")

joblib.dump(scaler, scaler_path)
print(f"スケーラー保存完了: {scaler_path}")

# === 予測詳細 CSV 出力 ===
pair_count = len(pred_t3)
with open(prediction_detail_csv_path, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerow(['時刻', '実測値', '予測値'])
    for i in range(pair_count):
        writer.writerow([i * 2 + 3, true_t3[i], pred_t3[i]])
        writer.writerow([i * 2 + 4, true_t4[i], pred_t4[i]])

# === 後続処理用変数 ===
pred_first  = pred_t3
true_first  = true_t3
pred_second = pred_t4
true_second = true_t4

# === 2サイクルごとの予測結果と大小関係をCSVに保存 ===
output_dir = os.path.dirname(csv_output_path)
os.makedirs(output_dir, exist_ok=True)

output_rows = []

for i in range(0, pair_count - 1, 1):  # 1ペアずつ（t3[i], t4[i]）
    cycle1 = i * 2 + 3
    cycle2 = i * 2 + 4
    pred1 = int(pred_t3[i])
    pred2 = int(pred_t4[i])

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
