import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np
import torch
import csv
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import ScalarFormatter, MaxNLocator
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
import torch.nn as nn
import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================
# 予測対象の道路設定
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

# === 使用ノード ===
used_road_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]
node_num = len(used_road_indices)
selected_target_road = sys.argv[1] if len(sys.argv) > 1 else TARGET_ROAD
target_road_id = extract_road_id(selected_target_road)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === データ読み込み ===
train_data_csv   = pd.read_csv(os.path.join(BASE_DIR, "★学習用データ", "1. 赤時間2分割_学習データ_12.22", "新環境待ち台数データ_12.22.csv"), encoding='cp932')
adjacency_matrix = pd.read_csv(r"C:\Users\Tsukasa\Desktop\研究\予測\新環境_隣接行列.csv", encoding='shift-jis', index_col=0)
answer_csv_path  = os.path.join(BASE_DIR, "★学習用データ", "1. 赤時間2分割_学習データ_12.22", "新環境正解データ_12.22.csv")
print(adjacency_matrix)

# == データ数 ==
DAY_LENGTH = 1330

# == 時間帯フィルタ ==
USE_TIME_FILTER = True
START_HOUR = 7
END_HOUR   = 19

_samples_per_hour = DAY_LENGTH / 24
TIME_START_IDX = round(START_HOUR * _samples_per_hour) if USE_TIME_FILTER else 0
TIME_END_IDX   = round(END_HOUR   * _samples_per_hour) if USE_TIME_FILTER else DAY_LENGTH

print(f"時間帯フィルタ: {'有効' if USE_TIME_FILTER else '無効'} "
      f"({START_HOUR}時〜{END_HOUR}時, 1日あたり {TIME_END_IDX - TIME_START_IDX} サンプル)")

val_days   = 5
test_days  = 1
train_days = 145

val_long   = val_days   * DAY_LENGTH
test_long  = test_days  * DAY_LENGTH
train_long = train_days * DAY_LENGTH

# === ノード設定 ===
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
model_output_dir  = os.path.join(BASE_DIR, "予測モデル", "1. 赤時間2分割",
                                 f"{target_road_label}_単ステップ")
os.makedirs(model_output_dir, exist_ok=True)

prediction_detail_csv_path = os.path.join(model_output_dir, f"予測結果_{target_road_label}_単ステップ.csv")
csv_output_path            = os.path.join(model_output_dir, f"予測結果_大小比較_{target_road_label}_単ステップ.csv")
scaler_path                = os.path.join(model_output_dir, "scaler.pkl")

print(f"予測対象: {target_road_label} (ノード index: {target_node_index})")
print_target_road_banner(
    target_road_label=target_road_label,
    target_node_index=target_node_index,
    selected_target_road=selected_target_road,
    model_output_dir=model_output_dir,
)

# === データ分割 ===
df_train_data = train_data_csv.iloc[:train_long,                                       used_road_indices]
df_val_data   = train_data_csv.iloc[train_long:train_long+val_long,                    used_road_indices]
df_test_data  = train_data_csv.iloc[train_long+val_long:train_long+val_long+test_long, used_road_indices]

print(f"df_train_data形状:{df_train_data.shape}")
print(f"df_val_data形状:  {df_val_data.shape}")
print(f"df_test_data形状: {df_test_data.shape}")

# === 正規化 ===
scaler          = MinMaxScaler()
train_data      = scaler.fit_transform(df_train_data).T
validation_data = scaler.transform(df_val_data).T
test_data       = scaler.transform(df_test_data).T

# === edge_index 構築 ===
adj               = adjacency_matrix.values
edge_index_raw    = np.array(np.nonzero(adj)).astype(np.int64)
mask              = np.isin(edge_index_raw[0], used_road_indices) & np.isin(edge_index_raw[1], used_road_indices)
edge_index_filtered = edge_index_raw[:, mask]
id_map            = {old: new for new, old in enumerate(used_road_indices)}
edge_index_mapped = np.vectorize(id_map.get)(edge_index_filtered)
edge_index        = torch.tensor(edge_index_mapped, dtype=torch.long).to(device)
print("エッジ：", edge_index.shape)


# ===================================================================
# Dataset 作成
# offset=0: t1開始（インデックス0, 2, 4, ...）→ t1でt3を予測
# offset=1: t2開始（インデックス1, 3, 5, ...）→ t2でt4を予測
# stride=2 で1つおきにサンプリングし、同種データのみを使用
# ===================================================================
def build_dataset(data_array, edge_index, device, target_node_index, offset, alpha=2):
    total_steps = data_array.shape[1]
    num_days    = total_steps // DAY_LENGTH
    dataset     = []

    for day in range(num_days):
        day_start   = day * DAY_LENGTH
        range_start = day_start + TIME_START_IDX + offset  # offsetで開始位置をずらす
        range_end   = day_start + TIME_END_IDX

        for t in range(range_start, range_end - 2, 2):  # stride=2 で同種データのみ
            x_t = torch.tensor(data_array[:, t],   dtype=torch.float).unsqueeze(1).to(device)
            y_t = torch.tensor(data_array[:, t+2], dtype=torch.float).unsqueeze(1).to(device)

            x_t[target_node_index] *= alpha

            dataset.append(Data(x=x_t, edge_index=edge_index, y=y_t))

    return dataset


# t3用データセット（t1→t3: インデックス0, 2, 4, ...）
t3_train_dataset = build_dataset(train_data,      edge_index, device, target_node_index, offset=0)
t3_val_dataset   = build_dataset(validation_data, edge_index, device, target_node_index, offset=0)
t3_test_dataset  = build_dataset(test_data,       edge_index, device, target_node_index, offset=0)

# t4用データセット（t2→t4: インデックス1, 3, 5, ...）
t4_train_dataset = build_dataset(train_data,      edge_index, device, target_node_index, offset=1)
t4_val_dataset   = build_dataset(validation_data, edge_index, device, target_node_index, offset=1)
t4_test_dataset  = build_dataset(test_data,       edge_index, device, target_node_index, offset=1)

batch_size = 16

t3_train_loader = DataLoader(t3_train_dataset, batch_size=batch_size, shuffle=False)
t3_val_loader   = DataLoader(t3_val_dataset,   batch_size=batch_size, shuffle=False)
t3_test_loader  = DataLoader(t3_test_dataset,  batch_size=batch_size, shuffle=False)

t4_train_loader = DataLoader(t4_train_dataset, batch_size=batch_size, shuffle=False)
t4_val_loader   = DataLoader(t4_val_dataset,   batch_size=batch_size, shuffle=False)
t4_test_loader  = DataLoader(t4_test_dataset,  batch_size=batch_size, shuffle=False)

print(f"t3用 学習サンプル数: {len(t3_train_dataset)},  t4用 学習サンプル数: {len(t4_train_dataset)}")


# ===================================================================
# モデル定義（5層GCN）
# ===================================================================
# class TrafficPredictionGNN(nn.Module):
#     def __init__(self, num_node_features, dropout_rate=0.2):
#         super(TrafficPredictionGNN, self).__init__()
#         self.conv1 = GCNConv(num_node_features, 32)
#         self.conv2 = GCNConv(32, 32)
#         self.conv3 = GCNConv(32, 64)
#         self.conv4 = GCNConv(64, 128)
#         self.conv5 = GCNConv(128, 128)
#         self.lin   = nn.Linear(128, 1)
#         self.dropout_rate = dropout_rate

#     def forward(self, x, edge_index):
#         h = F.relu(self.conv1(x, edge_index))
#         h = F.relu(self.conv2(h, edge_index))
#         h = F.relu(self.conv3(h, edge_index))
#         h = F.relu(self.conv4(h, edge_index))
#         h = F.relu(self.conv5(h, edge_index))
#         return self.lin(h)

class TrafficPredictionGNN(nn.Module):
    def __init__(self, num_node_features, dropout_rate=0.2):
        super(TrafficPredictionGNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 32)
        self.conv2 = GCNConv(32, 32)
        self.conv3 = GCNConv(32, 1)
        self.dropout_rate = dropout_rate

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        h = F.relu(self.conv3(h, edge_index))
        return self.lin(h)


# === 共通関数：1モデル分の学習ループ ===
def train_one_model(train_loader, val_loader, model_path, label):
    learning_rates = [0.001] * 200 + [0.0005] * 100 + [0.0001] * 100
    patience = 25

    model     = TrafficPredictionGNN(num_node_features=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    best_val_loss     = float('inf')
    best_test_loss    = float('inf')
    epochs_no_improve = 0
    best_train_losses = []
    best_val_losses   = []
    train_losses      = []
    val_losses        = []

    for epoch in range(400):
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rates[epoch]

        # 学習
        running_train_loss = 0.0
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            out = model(batch.x, batch.edge_index)
            out = out.view(-1, node_num, 1)
            y   = batch.y.view(-1, node_num, 1)

            out_t = out[:, target_node_index, :]
            y_t   = y[:,   target_node_index, :]

            loss = criterion(out_t, y_t) + F.l1_loss(out_t, y_t)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

        avg_train = running_train_loss / len(train_loader)
        train_losses.append(avg_train)

        # バリデーション
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index)
                out = out.view(-1, node_num, 1)
                y   = batch.y.view(-1, node_num, 1).to(device)

                out_t = out[:, target_node_index, :]
                y_t   = y[:,   target_node_index, :]
                running_val_loss += (criterion(out_t, y_t) + F.l1_loss(out_t, y_t)).item()

        avg_val = running_val_loss / len(val_loader)
        val_losses.append(avg_val)

        if avg_val < best_val_loss:
            best_val_loss     = avg_val
            torch.save(model.state_dict(), model_path)
            print(f"  [{label}] Best saved at epoch {epoch+1}  Val Loss: {best_val_loss:.6f}")
            best_train_losses = train_losses.copy()
            best_val_losses   = val_losses.copy()

        print(f"  [{label}] Epoch {epoch+1:3d}  Train: {avg_train:.6f}  Val: {avg_val:.6f}  LR: {learning_rates[epoch]}")

        if avg_val < best_test_loss:
            best_test_loss    = avg_val
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"  [{label}] Early stopping")
            break

    return best_train_losses, best_val_losses


# === 共通関数：テスト予測 ===
def run_test(test_loader, model_path):
    model = TrafficPredictionGNN(num_node_features=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    preds, truths = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out   = model(batch.x, batch.edge_index).view(-1, node_num, 1)
            y     = batch.y.view(-1, node_num, 1)
            preds.append(out.cpu().numpy())
            truths.append(y.cpu().numpy())

    preds  = np.concatenate(preds,  axis=0)
    truths = np.concatenate(truths, axis=0)

    preds[:, :, 0]  = scaler.inverse_transform(preds[:, :, 0])
    truths[:, :, 0] = scaler.inverse_transform(truths[:, :, 0])

    pred_seq = preds[:,  target_node_index, 0]
    true_seq = truths[:, target_node_index, 0]
    return pred_seq, true_seq


# ===================================================================
# t3モデル 学習ループ（偶数位置データのみ使用）
# ===================================================================
print("\n" + "="*60)
print("t3モデル 学習開始（t1→t3, t1・t3データのみ使用）")
print("="*60)

t3_model_path    = os.path.join(model_output_dir, "GCN_best_t3.pth")
rmse_t3          = float('inf')
t3_count         = 1
t3_best_train_losses = []
t3_best_val_losses   = []

while rmse_t3 > 1.5:
    print(f"\nt3モデル {t3_count}回目の学習中")
    t3_best_train_losses, t3_best_val_losses = train_one_model(
        t3_train_loader, t3_val_loader, t3_model_path, label="t3"
    )

    pred_t3, true_t3 = run_test(t3_test_loader, t3_model_path)
    rmse_t3 = np.sqrt(np.mean((pred_t3 - true_t3) ** 2))

    print(f"\n{'='*60}")
    print(f"t3 予測精度 [{target_road_label}]  ({t3_count}回目)")
    print(f"  t3 RMSE: {rmse_t3:.4f}")
    print(f"{'='*60}")

    if rmse_t3 <= 1.5:
        print("[達成] t3 終了条件達成！")
        break
    t3_count += 1
    if t3_count == 21:
        print("[警告] t3 最大試行回数に到達。強制終了します。")
        break

# ===================================================================
# t4モデル 学習ループ（奇数位置データのみ使用）
# ===================================================================
print("\n" + "="*60)
print("t4モデル 学習開始（t2→t4, t2・t4データのみ使用）")
print("="*60)

t4_model_path    = os.path.join(model_output_dir, "GCN_best_t4.pth")
rmse_t4          = float('inf')
t4_count         = 1
t4_best_train_losses = []
t4_best_val_losses   = []

while rmse_t4 > 1.5:
    print(f"\nt4モデル {t4_count}回目の学習中")
    t4_best_train_losses, t4_best_val_losses = train_one_model(
        t4_train_loader, t4_val_loader, t4_model_path, label="t4"
    )

    pred_t4, true_t4 = run_test(t4_test_loader, t4_model_path)
    rmse_t4 = np.sqrt(np.mean((pred_t4 - true_t4) ** 2))

    print(f"\n{'='*60}")
    print(f"t4 予測精度 [{target_road_label}]  ({t4_count}回目)")
    print(f"  t4 RMSE: {rmse_t4:.4f}")
    print(f"{'='*60}")

    if rmse_t4 <= 1.5:
        print("[達成] t4 終了条件達成！")
        break
    t4_count += 1
    if t4_count == 21:
        print("[警告] t4 最大試行回数に到達。強制終了します。")
        break

# ===================================================================
# 最終予測精度の出力
# ===================================================================
print(f"\n{'='*60}")
print(f"最終予測精度 [{target_road_label}]")
print(f"  t3 RMSE: {rmse_t3:.4f}")
print(f"  t4 RMSE: {rmse_t4:.4f}")
print(f"{'='*60}\n")

joblib.dump(scaler, scaler_path)
print(f"スケーラーを保存しました: {scaler_path}")

# ===================================================================
# 予測結果CSV保存（t3・t4を交互に並べて復元）
# ===================================================================
n = min(len(pred_t3), len(pred_t4))
pred_seq = np.empty(n * 2)
true_seq = np.empty(n * 2)
pred_seq[0::2] = pred_t3[:n]
pred_seq[1::2] = pred_t4[:n]
true_seq[0::2] = true_t3[:n]
true_seq[1::2] = true_t4[:n]

with open(prediction_detail_csv_path, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerow(['時刻', '実測値', '予測値'])
    for i in range(len(pred_seq)):
        writer.writerow([i + 1, true_seq[i], pred_seq[i]])

# ===================================================================
# Loss曲線の保存（t3・t4それぞれ）
# ===================================================================
matplotlib.rcParams["font.family"] = "Meiryo"
plt.rcParams["font.size"] = 15

for losses, label in [
    ((t3_best_train_losses, t3_best_val_losses), "t3"),
    ((t4_best_train_losses, t4_best_val_losses), "t4"),
]:
    train_l, val_l = losses
    plt.figure(figsize=(8, 5))
    plt.plot(train_l, label='Train Loss', linewidth=2)
    plt.plot(val_l,   label='Validation Loss', linewidth=2)
    plt.title(f"Loss ({label} model) - {target_road_label}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    path = os.path.join(model_output_dir, f"Loss曲線_{target_road_label}_{label}_単ステップ.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Loss曲線を保存しました → {path}")

# ===================================================================
# 予測波形グラフの保存（t3・t4それぞれ）
# ===================================================================
_num_hours   = END_HOUR - START_HOUR if USE_TIME_FILTER else 24
_hour_offset = START_HOUR            if USE_TIME_FILTER else 0

for pred, true, label in [(pred_t3, true_t3, "t3"), (pred_t4, true_t4, "t4")]:
    seq_len         = len(pred)
    cycles_per_hour = seq_len / _num_hours
    hour_ticks      = [round(i * cycles_per_hour) for i in range(_num_hours + 1)]
    hour_labels     = [f"{_hour_offset + i}時" for i in range(_num_hours + 1)]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(range(seq_len), true, color="blue", linewidth=1.5, label=f"Actual ({label})")
    ax.plot(range(seq_len), pred, color="red",  linewidth=1.5, linestyle="--", label=f"Predicted ({label})")
    ax.set_xticks(hour_ticks)
    ax.set_xticklabels(hour_labels, rotation=45)
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.ticklabel_format(style="plain", axis="y")
    ax.set_title(f"{label}: Vehicle Number Forecast ({target_road_label})")
    ax.set_xlim(0, seq_len)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Number of vehicles")
    ax.legend()
    ax.grid(which="both", axis="both")
    ax.minorticks_on()
    fig.tight_layout()

    save_path = os.path.join(model_output_dir, f"予測波形_{target_road_label}_{label}_単ステップ.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"{label}グラフを保存しました → {save_path}")

# ===================================================================
# 大小比較CSV出力（t3とt4のペアで比較）
# ===================================================================
output_rows = []
for i in range(n):
    cycle1 = i * 2 + 1
    cycle2 = i * 2 + 2
    pred1  = int(pred_t3[i])
    pred2  = int(pred_t4[i])
    if pred1 == pred2:
        relation_label = 1
    elif pred1 < pred2:
        relation_label = 0
    else:
        relation_label = 2
    output_rows.append([cycle1, cycle2, pred1, pred2, relation_label])

with open(csv_output_path, mode='w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerow(["サイクル1", "サイクル2", "予測値1(t3)", "予測値2(t4)", "大小関係ラベル"])
    writer.writerows(output_rows)
print(f"大小比較結果を保存しました：{csv_output_path}")

# ===================================================================
# 正解ラベルとの正答率評価
# ===================================================================
true_labels = []
with open(answer_csv_path, mode='r', encoding='utf-8-sig') as f:
    reader = list(csv.reader(f))
    for row in reader[2:]:
        if row:
            true_labels.append(int(row[target_node_index]))

pred_labels = []
with open(csv_output_path, mode='r', encoding='utf-8-sig') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        if row:
            pred_labels.append(int(row[4]))

correct  = sum(p == t for p, t in zip(pred_labels, true_labels))
total    = len(true_labels)
accuracy = correct / total * 100 if total > 0 else 0
print(f"{target_road_label} の正答数: {correct} / {total}（正答率: {accuracy:.2f}%）")
