import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

# ============================================================================
# 道路1の「待ち台数」と「最大間隔発生台数目」を同時予測するモデル（単一ステップ版）
# ----------------------------------------------------------------------------
# ★学習器の機構（GCNConv×3 + スキップ接続 / 学習率スケジュール / early stopping /
#   RMSE 未達なら再学習する while ループ）は元コードと同一。
#
# ★予測構造（前半/後半を廃止 → 1ステップ先のみ予測）:
#     各道路の「一つの値(t)」から「次の値(t+1)」を予測する。
#       入力 : ノード特徴量 [待ち台数(t), 間隔台数目(t)]   (num_node_features = 2)
#               待ち台数 は道路1ノードのみ ×ALPHA / 間隔台数目 は道路1ノードのみ値
#       出力 : 道路1の [待ち台数(t+1), 間隔台数目(t+1)]    (2値)
#     → t3/t4 の 2 ステップ自己回帰・Teacher Forcing・Scheduled Sampling は廃止。
#     → RMSE も前半/後半に分けず、待ち台数・間隔台数目それぞれ単一値で評価。
#
# ★入力データ:
#     待ち台数_学習データ.csv          : 全道路(道路1〜9)の待ち台数  [N, 9]
#     最大間隔発生台数目_学習データ.csv : 道路1の最大間隔発生台数目   [N, 1]
# ★データ量: 全 181 日分 × 1 日 720 サンプル = 130320 行
# ============================================================================

import pandas as pd
import numpy as np
import torch
import csv
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from torch_geometric.data import Data
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
from matplotlib.ticker import ScalarFormatter, MaxNLocator
import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================
# 予測対象の道路設定（このモデルは道路1固定: 間隔台数目が道路1のものだけ）
# =========================
TARGET_ROAD = "道路1"


def extract_road_id(value, fallback=None):
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    if digits:
        return int(digits)
    if fallback is not None:
        return fallback
    raise ValueError(f"道路番号を判定できません: {value}")


print(torch.cuda.is_available())
print(f"TARGET_ROAD 設定値: {TARGET_ROAD}")

# === 使用ノード（道路1〜9 すべて） ===
used_road_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]
node_num = len(used_road_indices)
selected_target_road = sys.argv[1] if len(sys.argv) > 1 else TARGET_ROAD
target_road_id = extract_road_id(selected_target_road)

# === デバイスの設定 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === データパス（★学習用データ フォルダ） ===
# 間隔台数目が「道路1の1列（ヘッダ = 最大間隔発生台数目）」になっているのは 2. のフォルダ。
DATA_DIR = os.path.join(BASE_DIR, "★学習用データ", "2. 到着間隔_学習データ_20260628 (道路1)")
wait_csv_path     = os.path.join(DATA_DIR, "待ち台数_学習データ.csv")            # 全道路の待ち台数 (9列)
interval_csv_path = os.path.join(DATA_DIR, "最大間隔発生台数目_学習データ.csv")  # 道路1の間隔台数目 (1列)
adjacency_path    = os.path.join(BASE_DIR, "新環境_隣接行列.csv")

# === データ読み込み ===
#   待ち台数 CSV は cp932 (Shift-JIS), ヘッダ = 道路1〜道路9
#   間隔台数目 CSV は utf-8-sig (BOM 付き), ヘッダ = 最大間隔発生台数目
wait_data_csv     = pd.read_csv(wait_csv_path,     encoding="cp932")
interval_data_csv = pd.read_csv(interval_csv_path, encoding="utf-8-sig")
adjacency_matrix  = pd.read_csv(adjacency_path,    encoding="shift-jis", index_col=0)
print(adjacency_matrix)

if len(wait_data_csv) != len(interval_data_csv):
    raise ValueError(
        f"待ち台数({len(wait_data_csv)}行)と間隔台数目({len(interval_data_csv)}行)の行数が一致しません"
    )

# == データ数 ==============================================================
DAY_LENGTH = 720  # 1日のサンプル数（130320 = 181日 × 720）
TOTAL_DAYS = len(wait_data_csv) // DAY_LENGTH
print(f"総行数: {len(wait_data_csv)} / DAY_LENGTH={DAY_LENGTH} → {TOTAL_DAYS} 日分")

# == 時間帯フィルタ設定 =====================================================
# True : 指定した時間帯のデータのみで学習 / False : 1日全体(0〜24時)を使用
#  ※ 元コードと同じく既定 True(10〜22時)。新データの時間構造が違う場合は要調整。
USE_TIME_FILTER = True
START_HOUR = 0   # 学習に使う開始時刻（0〜23）
END_HOUR   = 24   # 学習に使う終了時刻（1〜24、START_HOUR より大きい値）

_samples_per_hour = DAY_LENGTH / 24
TIME_START_IDX = round(START_HOUR * _samples_per_hour) if USE_TIME_FILTER else 0
TIME_END_IDX   = round(END_HOUR   * _samples_per_hour) if USE_TIME_FILTER else DAY_LENGTH

print(f"時間帯フィルタ: {'有効' if USE_TIME_FILTER else '無効'} "
      f"({START_HOUR}時〜{END_HOUR}時, 1日あたり {TIME_END_IDX - TIME_START_IDX} サンプル)")
# ==========================================================================

# === train / val / test の日数（合計 181 日に収める） ===
val_days   = 10
test_days  = 1
train_days = TOTAL_DAYS - val_days - test_days   # = 175

val_long   = val_days   * DAY_LENGTH
test_long  = test_days  * DAY_LENGTH
train_long = train_days * DAY_LENGTH
print(f"日数構成: train={train_days}日 / val={val_days}日 / test={test_days}日")

# === 損失と入力に関するハイパーパラメータ ===
ALPHA        = 2     # 道路1ノードの待ち台数特徴量を α 倍して強調（元コードと同じ）
LAMBDA_INT   = 1.0   # 損失における 間隔台数目 の重み
WAIT_RMSE_TH = 1.5   # 待ち台数 RMSE の終了しきい値
INT_RMSE_TH  = 1.5   # 間隔台数目 RMSE の終了しきい値
MAX_TRIALS   = 25    # 再学習の最大試行回数

# === 対象ノードの特定（道路1 = ノード0） ===
used_road_labels = [str(wait_data_csv.columns[i]) for i in used_road_indices]
used_road_ids = [
    extract_road_id(label, used_road_indices[pos] + 1)
    for pos, label in enumerate(used_road_labels)
]
if target_road_id not in used_road_ids:
    raise ValueError(f"対象道路{target_road_id}は使用道路に含まれていません。選択可能: {used_road_ids}")

target_node_index = used_road_ids.index(target_road_id)
target_road_label = used_road_labels[target_node_index]

# 成果物はすべてこのフォルダに出力する。
# ※ ★予測モデル作成_到着間隔込み.py（分布形状損失あり・道路1/5対応）とは
#    別フォルダに出力し、互いに上書きしないようにしている。
model_output_dir = os.path.join(BASE_DIR, "予測モデル", "2. 到着間隔_同時予測",
                                f"{target_road_label}_旧版")
os.makedirs(model_output_dir, exist_ok=True)

best_model_path   = os.path.join(model_output_dir, "GCN_epoch_best_model.pth")
scaler_wait_path  = os.path.join(model_output_dir, "scaler_wait.pkl")
scaler_int_path   = os.path.join(model_output_dir, "scaler_interval.pkl")

pred_wait_csv_path = os.path.join(model_output_dir, f"予測結果_{target_road_label}_待ち台数.csv")
pred_int_csv_path  = os.path.join(model_output_dir, f"予測結果_{target_road_label}_間隔台数目.csv")
loss_plot_path     = os.path.join(model_output_dir, f"Loss曲線_{target_road_label}_同時予測.png")

print(f"予測対象: {target_road_label} (ノード index: {target_node_index})")
print(f"保存先   : {model_output_dir}")

# === train / val / test 分割 ===
# 待ち台数: 9 列, 間隔台数目: 1 列
df_train_wait = wait_data_csv.iloc[:train_long, used_road_indices]
df_val_wait   = wait_data_csv.iloc[train_long:train_long + val_long, used_road_indices]
df_test_wait  = wait_data_csv.iloc[train_long + val_long:train_long + val_long + test_long, used_road_indices]

df_train_int = interval_data_csv.iloc[:train_long, [0]]
df_val_int   = interval_data_csv.iloc[train_long:train_long + val_long, [0]]
df_test_int  = interval_data_csv.iloc[train_long + val_long:train_long + val_long + test_long, [0]]

print(f"df_train_wait形状:{df_train_wait.shape} / df_train_int形状:{df_train_int.shape}")

# === 正規化（スケーラー 2 個） ===
scaler_wait = MinMaxScaler()
scaler_int  = MinMaxScaler()

# 待ち台数: [time, 9] → fit → 転置して [9, time]（元コードと同じ向き）
train_wait = scaler_wait.fit_transform(df_train_wait).T
val_wait   = scaler_wait.transform(df_val_wait).T
test_wait  = scaler_wait.transform(df_test_wait).T

# 間隔台数目: [time, 1] → fit → 1次元 [time] に
train_int = scaler_int.fit_transform(df_train_int).reshape(-1)
val_int   = scaler_int.transform(df_val_int).reshape(-1)
test_int  = scaler_int.transform(df_test_int).reshape(-1)

print("=== 待ち台数 MinMaxScaler 範囲 ===")
for i, (mn, mx) in enumerate(zip(scaler_wait.data_min_, scaler_wait.data_max_)):
    print(f"  ノード{i}（{used_road_labels[i]}）: min={mn}, max={mx}")
print(f"=== 間隔台数目 MinMaxScaler 範囲 === min={scaler_int.data_min_[0]}, max={scaler_int.data_max_[0]}")

# 逆正規化ヘルパ（道路1 の列のみ）
w_min0, w_max0 = scaler_wait.data_min_[target_node_index], scaler_wait.data_max_[target_node_index]
i_min,  i_max  = scaler_int.data_min_[0], scaler_int.data_max_[0]
def inv_wait(x):
    return x * (w_max0 - w_min0) + w_min0
def inv_int(x):
    return x * (i_max - i_min) + i_min

# === edge_index 再構築 ===
adj = adjacency_matrix.values
edge_index_raw = np.array(np.nonzero(adj)).astype(np.int64)
mask = np.isin(edge_index_raw[0], used_road_indices) & np.isin(edge_index_raw[1], used_road_indices)
edge_index_filtered = edge_index_raw[:, mask]
id_map = {old: new for new, old in enumerate(used_road_indices)}
edge_index_mapped = np.vectorize(id_map.get)(edge_index_filtered)
edge_index = torch.tensor(edge_index_mapped, dtype=torch.long).to(device)
print("エッジ：", edge_index.shape)


# === Dataset 作成（日ごとに構成・単一ステップ） =============================
# 各サンプル:「一つの値(t)」から「次の値(t+1)」を予測
#   x : [9, 2]  ノード特徴量 [待ち台数(t), 間隔台数目(t)]
#               待ち台数 は道路1ノードのみ ×ALPHA / 間隔台数目 は道路1ノードのみ値
#   y : [1, 2]  道路1の [待ち台数(t+1), 間隔台数目(t+1)]
def build_daywise_dataset(wait_arr, int_arr, edge_index, device, target_node_index, alpha=ALPHA):
    total_steps = wait_arr.shape[1]
    num_days = total_steps // DAY_LENGTH
    dataset = []

    for day in range(num_days):
        day_start = day * DAY_LENGTH
        range_start = day_start + TIME_START_IDX
        range_end   = day_start + TIME_END_IDX

        # t から t+1 を予測（t+1 が時間帯フィルタ内に収まる範囲）
        for t in range(range_start, range_end - 1):
            # --- 入力特徴量（時刻 t） ---
            w_t = torch.tensor(wait_arr[:, t], dtype=torch.float).unsqueeze(1)  # [9, 1]
            w_t[target_node_index] *= alpha
            i_t = torch.zeros((node_num, 1), dtype=torch.float)                 # 間隔は道路1のみ
            i_t[target_node_index, 0] = float(int_arr[t])
            x = torch.cat([w_t, i_t], dim=1).to(device)                         # [9, 2]

            # --- 目的変数（時刻 t+1, 道路1のみ） ---
            y = torch.tensor(
                [[wait_arr[target_node_index, t + 1], float(int_arr[t + 1])]],
                dtype=torch.float).to(device)                                  # [1, 2]

            dataset.append(Data(x=x, edge_index=edge_index, y=y))

    return dataset


train_dataset = build_daywise_dataset(train_wait, train_int, edge_index, device, target_node_index)
val_dataset   = build_daywise_dataset(val_wait,   val_int,   edge_index, device, target_node_index)
test_dataset  = build_daywise_dataset(test_wait,  test_int,  edge_index, device, target_node_index)
print(f"サンプル数 train={len(train_dataset)} / val={len(val_dataset)} / test={len(test_dataset)}")

# === バッチ作成 ===
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)


# ==== 単一ステップ予測モデル（待ち台数・間隔台数目の 2 出力） ====
# 一つの値(t) から 次の値(t+1) を予測。自己回帰・Teacher Forcing なし。
class DualPredictionGNN(torch.nn.Module):
    def __init__(self, num_node_features, dropout_rate=0.2):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, 128)
        self.skip  = nn.Linear(num_node_features, 128)   # スキップ接続

        self.lin_wait = nn.Linear(128, 1)   # 待ち台数(t+1)
        self.lin_int  = nn.Linear(128, 1)   # 間隔台数目(t+1)
        self.dropout_rate = dropout_rate

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        h = F.relu(self.conv3(h, edge_index))
        h = h + self.skip(x)                              # スキップ接続

        wait = self.lin_wait(h)                           # [num_nodes, 1]
        intv = self.lin_int(h)                            # [num_nodes, 1]
        return torch.cat([wait, intv], dim=1)             # [num_nodes, 2] = [待ち(t+1), 間隔(t+1)]


# 学習率スケジュール（合計 400 エポック・元コードと同じ）
learning_rates = [0.001] * 200 + [0.0005] * 100 + [0.0001] * 100

patience = 15  # early stopping

# === Loss履歴 ===
best_overall_train_losses = []
best_overall_val_losses = []


def dual_loss(out_target, y_target, criterion):
    """ out_target: [B,2], y_target: [B,2]  各列 = [待ち台数, 間隔台数目]。
        各量について MSE + MAE。"""
    mse_w = criterion(out_target[:, 0:1], y_target[:, 0:1])
    mae_w = F.l1_loss(out_target[:, 0:1], y_target[:, 0:1])
    loss_wait = mse_w + mae_w
    mse_i = criterion(out_target[:, 1:2], y_target[:, 1:2])
    mae_i = F.l1_loss(out_target[:, 1:2], y_target[:, 1:2])
    loss_int = mse_i + mae_i
    return loss_wait + LAMBDA_INT * loss_int


# RMSE 初期化（両方とも未達状態から開始）
rmse_wait = float('inf')
rmse_int  = float('inf')
prediction_count = 1

# ============================================================================
# 学習ループ（待ち台数・間隔台数目の RMSE が両方しきい値以下になるまで再学習）
# ============================================================================
while (rmse_wait > WAIT_RMSE_TH or rmse_int > INT_RMSE_TH):

    model = DualPredictionGNN(num_node_features=2, dropout_rate=0.2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    train_losses, val_losses = [], []
    epochs_no_improve = 0
    best_val_loss = float('inf')
    best_test_loss = float('inf')

    print(f"{prediction_count}回目の学習中 [{target_road_label} 同時予測・単一ステップ]")
    for epoch in range(400):
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rates[epoch]

        # --- 学習 ---
        running_train_loss = 0.0
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            out = out.view(-1, node_num, 2)
            out_target = out[:, target_node_index, :]    # [B, 2]
            y_target   = batch.y                          # [B, 2]

            loss = dual_loss(out_target, y_target, criterion)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # --- 検証 ---
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index)
                out = out.view(-1, node_num, 2)
                out_target = out[:, target_node_index, :]
                y_target   = batch.y
                loss = dual_loss(out_target, y_target, criterion)
                running_val_loss += loss.item()

        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # --- ベストモデル保存 ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved at epoch {epoch+1} with Val Loss: {best_val_loss}")
            best_overall_train_losses = train_losses.copy()
            best_overall_val_losses = val_losses.copy()

        print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss:.6f}, '
              f'Val Loss: {avg_val_loss:.6f}, 学習率: {learning_rates[epoch]}')

        # --- Early Stopping ---
        if avg_val_loss < best_test_loss:
            best_test_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print('Early stopping')
            break

    # === テスト予測 ===
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    preds, ys = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            out = out.view(-1, node_num, 2)[:, target_node_index, :]   # [b, 2]
            preds.append(out.cpu().numpy())
            ys.append(batch.y.cpu().numpy())

    preds = np.concatenate(preds, axis=0)   # [N, 2]
    ys    = np.concatenate(ys, axis=0)      # [N, 2]

    # === 逆正規化（道路1） ===
    pred_w = inv_wait(preds[:, 0]); pred_i = inv_int(preds[:, 1])
    true_w = inv_wait(ys[:, 0]);    true_i = inv_int(ys[:, 1])

    # === RMSE（前半/後半なし・単一値） ===
    rmse_wait = np.sqrt(np.mean((pred_w - true_w) ** 2))
    rmse_int  = np.sqrt(np.mean((pred_i - true_i) ** 2))
    print(f"[待ち台数]   RMSE: {rmse_wait:.4f}")
    print(f"[間隔台数目] RMSE: {rmse_int:.4f}")

    # === 予測結果 CSV 出力 ===
    with open(pred_wait_csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['サンプル', '実測値', '予測値'])
        for k in range(len(preds)):
            writer.writerow([k + 1, true_w[k], pred_w[k]])

    with open(pred_int_csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['サンプル', '実測値', '予測値'])
        for k in range(len(preds)):
            writer.writerow([k + 1, true_i[k], pred_i[k]])

    # === スケーラー保存 ===
    joblib.dump(scaler_wait, scaler_wait_path)
    joblib.dump(scaler_int,  scaler_int_path)
    print(f"スケーラー保存: {scaler_wait_path} / {scaler_int_path}")

    # === 終了条件（両方の RMSE がしきい値以下） ===
    if rmse_wait <= WAIT_RMSE_TH and rmse_int <= INT_RMSE_TH:
        print(f"[達成] 終了条件達成！ 待ち台数RMSE={rmse_wait:.3f} / 間隔RMSE={rmse_int:.3f}")
        break
    else:
        print(f"[継続] 再学習します（試行 {prediction_count}/{MAX_TRIALS}）")

    prediction_count += 1
    if prediction_count > MAX_TRIALS:
        print("[警告] 最大試行回数に到達。強制終了します。")
        break

# === ベストモデルの Loss 曲線をプロット ===
plt.figure(figsize=(8, 5))
plt.plot(best_overall_train_losses, label='Train Loss', linewidth=2)
plt.plot(best_overall_val_losses, label='Validation Loss', linewidth=2)
plt.title("Training & Validation Loss (Best Model Cycle)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Loss 曲線を保存しました → {loss_plot_path}")


# === 予測波形グラフ（待ち台数 / 間隔台数目） ===
import matplotlib
matplotlib.rcParams["font.family"] = "Meiryo"
plt.rcParams["font.size"] = 15

if USE_TIME_FILTER:
    _num_hours, _hour_offset = END_HOUR - START_HOUR, START_HOUR
else:
    _num_hours, _hour_offset = 24, 0


def plot_waveform(true_series, pred_series, title, ylabel, save_path):
    n = len(true_series)
    cph = n / _num_hours
    ticks = [round(i * cph) for i in range(_num_hours + 1)]
    labels = [f"{_hour_offset + i}時" for i in range(_num_hours + 1)]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(range(n), true_series, color="blue", linewidth=1.5, label="Actual")
    ax.plot(range(n), pred_series, color="red", linewidth=1.5, linestyle="--", label="Predicted")
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=45)
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.ticklabel_format(style="plain", axis="y")
    ax.set_title(title)
    ax.set_xlim(0, n)
    ax.set_xlabel("Cycle")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(which="both", axis="both")
    ax.minorticks_on()
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"グラフを保存しました → {save_path}")


plot_waveform(true_w, pred_w, f"待ち台数 ({target_road_label})", "待ち台数",
              os.path.join(model_output_dir, f"予測波形_{target_road_label}_待ち台数.png"))
plot_waveform(true_i, pred_i, f"最大間隔発生台数目 ({target_road_label})", "台数目",
              os.path.join(model_output_dir, f"予測波形_{target_road_label}_間隔台数目.png"))

print("=== 完了 ===")
print(f"待ち台数   RMSE: {rmse_wait:.4f}")
print(f"間隔台数目 RMSE: {rmse_int:.4f}")
