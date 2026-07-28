import sys
import os
sys.stdout.reconfigure(encoding='utf-8')

os.environ['SUMO_HOME'] = r'C:\Program Files (x86)\Eclipse\Sumo'
sys.path.append(r'C:\Program Files (x86)\Eclipse\Sumo\tools')

import csv
import math
import re
import xml.etree.ElementTree as ET
import torch
import traci
import sumolib
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
import warnings
import joblib


class TrafficPredictionGNN(torch.nn.Module):
    def __init__(self, num_node_features, dropout_rate=0.2):
        super(TrafficPredictionGNN, self).__init__()
        from torch_geometric.nn import GCNConv
        self.conv1 = GCNConv(num_node_features, 32)
        self.conv2 = GCNConv(32, 32)
        self.conv3 = GCNConv(32, 64)
        self.conv4 = GCNConv(64, 128)
        self.conv5 = GCNConv(128, 128)
        self.lin = torch.nn.Linear(128, 2)

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.relu(self.conv3(x, edge_index))
        x = torch.relu(self.conv4(x, edge_index))
        x = torch.relu(self.conv5(x, edge_index))
        out = self.lin(x)
        return torch.relu(out)


shown_warnings = set()

def show_warning_once(message):
    if message not in shown_warnings:
        warnings.warn(message, FutureWarning)
        shown_warnings.add(message)

def get_time_label_and_step(step, total_steps_in_day=1330, num_classes=5, prep_time=1300):
    if step < prep_time:
        raise ValueError(f"❌ step={step} は準備時間 {prep_time} 秒未満です。時間ラベルは付与されません。")
    step_in_day = step % total_steps_in_day
    block_size = total_steps_in_day // num_classes
    time_label = step_in_day // block_size
    return time_label, step_in_day


# ============================================================
# 汎用予測関数（道路1・道路5 両方に対応）
# ============================================================
def run_prediction(t1_array, delay_array, edge_index, scaler, model_path, prediction_count, num_node_features=2):
    """
    学習時と一致する、時間ラベルなしのリアルタイム予測関数。
    model_path と scaler を差し替えることで道路1・道路5 どちらにも対応。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_t  = np.array(t1_array).reshape(1, -1)
    x_t1 = np.array(delay_array).reshape(1, -1)
    x_t  = scaler.transform(x_t).T   # [ノード数, 1]
    x_t1 = scaler.transform(x_t1).T

    x_t  = torch.tensor(x_t,  dtype=torch.float).to(device)
    x_t1 = torch.tensor(x_t1, dtype=torch.float).to(device)

    x_input = torch.cat([x_t, x_t1], dim=1).to(device)  # [ノード数, 2]

    model = TrafficPredictionGNN(num_node_features=num_node_features).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        out = model(x_input, edge_index.to(device)).cpu().numpy()  # [ノード数, 2]

    for i in range(2):
        out[:, i] = scaler.inverse_transform(out[:, i].reshape(1, -1))[0]

    if (out < 0).any():
        print("⚠️ 予測結果にマイナス値があります。修正推奨")
        neg_idx = np.argwhere(out < 0)
        for node, dim in neg_idx:
            print(f"   ノード{node}・次元{dim}: {out[node, dim]:.3f}")

    return out[0]  # [t3, t4]


# ============================================================
# ログ出力クラス
# ============================================================
class TrafficLogger:
    def __init__(self, used_road_ids):
        self.used_road_ids = used_road_ids
        self.measurement_count = 0
        self.prediction_count = 0

    def log_measurement_data(self, road_id, t1_value, t2_minus_t1_value, step):
        self.measurement_count += 1

    def log_prediction_trigger(self, all_data, label=""):
        print(f"🔍 === 予測実行時の全道路データ {label}===")
        for road_id in self.used_road_ids:
            if all_data[road_id] is not None:
                t1, t2_minus_t1 = all_data[road_id]
                print(f"   道路{road_id}: t1={t1:2d}, t2-t1={t2_minus_t1:2d}")
            else:
                print(f"   道路{road_id}: データなし")
        print("=====================================")

    def log_prediction_result(self, count, t1_count, diff, pred_t3, pred_t4, step, label="道路1"):
        print(f"🔮 [{label}][{count:3d}回目] 入力: t1={t1_count:2d}, t2-t1={diff:2d} → 予測: t3={pred_t3:2d}, t4={pred_t4:2d} (step={step})")

    def log_data_incomplete(self, step, label=""):
        print(f"⏭️ {label}予測パス（step={step}）: データ未揃い → 破棄")


# ============================================================
# パス設定
# ============================================================
sumocfg_path  = r"C:\Users\Tsukasa\Desktop\研究\予測\町モデルデータ\toyama_shouwa.sumocfg"
log_dir_path  = r"C:\Users\Tsukasa\Desktop\研究\制御\csv掃き出し\各道路計測ログ"
train_csv_path = r"C:\Users\Tsukasa\Desktop\研究\予測\新環境待ち台数データ_12.22.csv"
adj_path      = r"C:\Users\Tsukasa\Desktop\研究\予測\新環境_隣接行列.csv"

# --- 道路1 モデル ---
model_path_road1  = r"C:\Users\Tsukasa\Desktop\研究\予測\予測モデル\0. 制御用_待ち台数単体(3.22)\道路1\GCN_epoch_best_model.pth"
scaler_path_road1 = r"C:\Users\Tsukasa\Desktop\研究\予測\予測モデル\0. 制御用_待ち台数単体(3.22)\道路1\scaler.pkl"

# --- 道路5 モデル（★追加） ---
model_path_road5  = r"C:\Users\Tsukasa\Desktop\研究\予測\予測モデル\0. 制御用_待ち台数単体(3.22)\道路5\GCN_epoch_best_model.pth"
scaler_path_road5 = r"C:\Users\Tsukasa\Desktop\研究\予測\予測モデル\0. 制御用_待ち台数単体(3.22)\道路5\scaler.pkl"


# ============================================================
# 使用ノード・スケーラー・edge_index（道路1用・道路5用 共通）
# ============================================================
used_road_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]
used_road_ids     = [str(i + 1) for i in range(9)]   # 道路1～9

latest_data = {rid: None for rid in used_road_ids}

logger = TrafficLogger(used_road_ids)

# スケーラーは後で joblib.load するのでここでは構築しない
# （run_simulation 内で読み込む）

# edge_index 構築
df_all   = pd.read_csv(train_csv_path, encoding='cp932')
df_used  = df_all.iloc[:, used_road_indices]
_scaler_tmp = MinMaxScaler()
_scaler_tmp.fit(df_used.values)   # edge_index 構築のためだけに使用

adj_matrix        = pd.read_csv(adj_path, encoding='shift-jis', index_col=0).values
edge_index_raw    = np.array(np.nonzero(adj_matrix)).astype(np.int64)
mask              = np.isin(edge_index_raw[0], used_road_indices) & np.isin(edge_index_raw[1], used_road_indices)
edge_index_filtered = edge_index_raw[:, mask]
id_map            = {old: new for new, old in enumerate(used_road_indices)}
edge_index_mapped = np.vectorize(id_map.get)(edge_index_filtered)
edge_index        = torch.tensor(edge_index_mapped, dtype=torch.long)


# ============================================================
# 信号設定
# ============================================================
MAX_RECORD_COUNT = 720
PREP_TIME = 1200

TRAFFIC_LIGHT_CONFIG = {
    "A": [
    {"id": "11", "edges": ["-E43", "-E44", "-973892289#0"], "red": "rrryyyyrrryyyyy", "green": "rrrrrrrrrrrrrrr"},
    {"id": "12", "edges": ["E45"], "red": "yyyyrrrrryyyyrrrrr", "green": "rrrrrrrrrrrrrrr"},
    {"id": "13", "edges": ["E46"], "red": "yyyyrrrrryyyyrrrrr", "green": "rrrrrrrrrrrrrrr"},
    {"id": "14", "edges": ["-E48"], "red": "rrryyyyrrryyyyy", "green": "rrrrrrrrrrrrrrrr"}
    ],
    "B": [
    {"id": "15", "edges": ["-155398386#0"], "red": "rrryyyrrryyy", "green": "rrrrrrrrrrrr"},
    {"id": "16", "edges": ["-973892417#0"], "red": "yyyrrryyyrrr", "green": "rrrrrrrrrrrr"},
    {"id": "17", "edges": ["622541624#11"], "red": "yyyrrryyyrrr", "green": "rrrrrrrrrrrr"},
    {"id": "18", "edges": ["155389111"], "red": "rrryyyrrryyy", "green": "rrrrrrrrrrrr"}
    ],
    "C": [
    {"id": "19", "edges": ["E128"], "red": "rrrryyyrrryyy", "green": "rrrrrrrrrrrrr"},
    {"id": "20", "edges": ["-E130"], "red": "yyyyrrryyyrrr", "green": "rrrrrrrrrrrrr"},
    {"id": "21", "edges": ["-E129"], "red": "yyyyrrryyyrrr", "green": "rrrrrrrrrrrrr"},
    {"id": "22", "edges": ["-E131"], "red": "rrrryyyrrryyy", "green": "rrrrrrrrrrrrr"}
    ],
    "D": [
    {"id": "23", "edges": ["-543210000#1", "-E41"], "red": "rrryyyyrrryyyy", "green": "rrrrrrrrrrrrrr"},
    {"id": "24", "edges": ["155394941#2"], "red": "yyyrrrryyyrrrr", "green": "rrrrrrrrrrrrrr"},
    {"id": "25", "edges": ["155398386#1"], "red": "yyyrrrryyyrrrr", "green": "rrrrrrrrrrrrrr"},
    {"id": "26", "edges": ["973892290", "E44"], "red": "rrryyyyrrryyyy", "green": "rrrrrrrrrrrrrr"},
    ],
    "E": [
    {"id": "2", "edges": ["E40", "-E22"], "red": "rrryyyyrrryyyy", "green": "rrrrrrrrrrrrrr"},
    {"id": "27", "edges": ["-E27"], "red": "yyyrrrryyyrrrr", "green": "rrrrrrrrrrrrrr"},
    {"id": "28", "edges": ["E42", "E41"], "red": "rrryyyyrrryyyy", "green": "rrrrrrrrrrrrrr"},
    ],
    "F": [
    {"id": "3", "edges":  ["E3", "E1", "-E34", "E35"], "red": "rrryyyyrrryyyy", "green": "rrrrrrrrrrrrrr"},
    {"id": "29", "edges": ["155396812#0"], "red": "yyyrrrryyyrrrr", "green": "rrrrrrrrrrrrrr"},
    {"id": "30", "edges": ["-155397698#1"], "red": "yyyrrrryyyrrrr", "green": "rrrrrrrrrrrrrr"},
    {"id": "31", "edges": ["E26", "E22"], "red": "rrryyyyrrryyyy", "green": "rrrrrrrrrrrrrr"},
    ],
    "G": [
    {"id": "32", "edges": ["E62"], "red": "rrrryyyrrrryyy", "green": "rrrrrrrrrrrrrr"},
    {"id": "33", "edges": ["-E58", "-E59"], "red": "yyyyrrryyyyrrr", "green": "rrrrrrrrrrrrrr"},
    {"id": "34", "edges": ["E61", "E60"], "red": "yyyyrrryyyyrrr", "green": "rrrrrrrrrrrrrr"},
    {"id": "35", "edges": ["-E63"], "red": "rrrryyyrrrryyy", "green": "rrrrrrrrrrrrrr"},
    ],
    "H": [
    {"id": "36", "edges": ["155391024"], "red": "rrrryyyrrrryyy", "green": "rrrrrrrrrrrrrr"},
    {"id": "37", "edges": ["-E56"], "red": "yyyyrrryyyyrrr", "green": "rrrrrrrrrrrrrr"},
    {"id": "38", "edges": ["-155398822#23", "E59"], "red": "yyyyrrryyyyrrr", "green": "rrrrrrrrrrrrrr"},
    {"id": "39", "edges": ["155390837#1"], "red": "rrrryyyrrrryyy", "green": "rrrrrrrrrrrrrr"},
    ],
    "I": [
    {"id": "40", "edges": ["-155387788"], "red": "rrrryyyrrrryyy", "green": "rrrrrrrrrrrrrr"},
    {"id": "41", "edges": ["155398822#21"], "red": "yyyyrrryyyyrrr", "green": "rrrrrrrrrrrrrr"},
    {"id": "4",  "edges": ["E57"], "red": "yyyyrrryyyyrrr", "green": "rrrrrrrrrrrrrr"},
    {"id": "42", "edges": ["155387430"], "red": "rrrryyyrrrryyy", "green": "rrrrrrrrrrrrrr"},
    ],
    "J": [
    {"id": "1",  "edges": ["E24", "-E23", "-E33"], "red": "rrrryyyyrrrryyyy", "green": "rrrrrrrrrrrrrrrr"},
    {"id": "5",  "edges": ["E37", "E36"],           "red": "yyyyrrrryyyyrrrr", "green": "rrrrrrrrrrrrrrrr"},
    {"id": "6",  "edges": ["-E25", "-E69"],         "red": "yyyyrrrryyyyrrrr", "green": "rrrrrrrrrrrrrrrr"},
    {"id": "10", "edges": ["-E35", "E34", "-E1"],   "red": "rrrryyyyrrrryyyy", "green": "rrrrrrrrrrrrrrrr"}
    ],
    "K": [
    {"id": "43", "edges": ["E71"], "red": "rrrrrryy", "green": "rrrrrrrr"},
    {"id": "7",  "edges": ["155398822#14", "155398822#13"], "red": "yyyyyyrr", "green": "rrrrrrrr"},
    {"id": "44", "edges": ["E70"], "red": "yyyyyyrr", "green": "rrrrrrrr"}
    ],
    "L": [
    {"id": "45", "edges": ["E76"], "red": "rrrryyyrrrryyy", "green": "rrrrrrrrrrrrrr"},
    {"id": "46", "edges": ["-E74", "-E75"], "red": "yyyyrrryyyyrrr", "green": "rrrrrrrrrrrrrr"},
    {"id": "47", "edges": ["E73", "E72"], "red": "yyyyrrryyyyrrr", "green": "rrrrrrrrrrrrrr"},
    {"id": "48", "edges": ["E77"], "red": "rrrryyyrrrryyy", "green": "rrrrrrrrrrrrrr"}
    ],
    "M": [
    {"id": "8",  "edges": ["E31", "E30"], "red": "rrryyyyrrryyyy", "green": "rrrrrrrrrrrrrr"},
    {"id": "49", "edges": ["155390062#6"], "red": "yyyrrrryyyrrrr", "green": "rrrrrrrrrrrrrr"},
    {"id": "50", "edges": ["155389284#1"], "red": "yyyrrrryyyrrrr", "green": "rrrrrrrrrrrrrr"},
    {"id": "51", "edges": ["543210000#15", "-E32"], "red": "rrryyyyrrryyyy", "green": "rrrrrrrrrrrrrr"}
    ],
    "N": [
    {"id": "9",  "edges": ["E20", "E19"], "red": "rrryyyyrrryyyy", "green": "rrrrrrrrrrrrrr"},
    {"id": "52", "edges": ["155390603#0"], "red": "yyyrrrryyyrrrr", "green": "rrrrrrrrrrrrrr"},
    {"id": "53", "edges": ["-155385264#1"], "red": "yyyrrrryyyrrrr", "green": "rrrrrrrrrrrrrr"},
    {"id": "54", "edges": ["-E21"], "red": "rrryyyyrrryyyy", "green": "rrrrrrrrrrrrrr"}
    ],
    "O": [
    {"id": "55", "edges": ["E18"], "red": "rrryyyyrrryyyy", "green": "rrrrrrrrrrrrrr"},
    {"id": "56", "edges": ["-155389447#1"], "red": "yyyrrrryyyrrrr", "green": "rrrrrrrrrrrrrr"},
    {"id": "57", "edges": ["155387086#0"], "red": "yyyrrrryyyrrrr", "green": "rrrrrrrrrrrrrr"},
    {"id": "58", "edges": ["-E17", "E16"], "red": "rrryyyyrrryyyy", "green": "rrrrrrrrrrrrrr"}
    ],
    "P": [
    {"id": "59", "edges": ["E13", "E12"], "red": "rrryyyyrrryyyy", "green": "rrrrrrrrrrrrrr"},
    {"id": "60", "edges": ["-E14"], "red": "yyyrrrryyyrrrr", "green": "rrrrrrrrrrrrrr"},
    {"id": "61", "edges": ["-E15"], "red": "yyyrrrryyyrrrr", "green": "rrrrrrrrrrrrrr"},
    {"id": "62", "edges": ["-E18"], "red": "rrryyyyrrryyyy", "green": "rrrrrrrrrrrrrr"}
    ],
}

prediction_count = 0

# ============================================================
# ログファイル設定
# ============================================================

# === 道路1 予測結果ログ ===
prediction_log_path = r"C:\Users\Tsukasa\Desktop\研究\制御\csv掃き出し\予測結果_道路1.csv"
prediction_log_file = open(prediction_log_path, "w", newline="", encoding="utf-8-sig")
prediction_writer   = csv.writer(prediction_log_file)
prediction_writer.writerow(["回数", "step", "前半", "後半", "予測前半", "予測後半"])

# === 道路5 予測結果ログ（★追加） ===
prediction_log_path_road5 = r"C:\Users\Tsukasa\Desktop\研究\制御\csv掃き出し\予測結果_道路5.csv"
prediction_log_file_road5 = open(prediction_log_path_road5, "w", newline="", encoding="utf-8-sig")
prediction_writer_road5   = csv.writer(prediction_log_file_road5)
prediction_writer_road5.writerow(["回数", "step", "前半", "後半", "予測前半", "予測後半"])

# === フェーズ時間ログ ===
phase_log_path = r"C:\Users\Tsukasa\Desktop\研究\制御\csv掃き出し\信号A_フェーズ時間ログ.csv"
phase_log_file = open(phase_log_path, "w", newline="", encoding="utf-8-sig")
phase_writer   = csv.writer(phase_log_file)
phase_writer.writerow(["回数", "step", "パターンA(秒)", "パターンB(秒)", "予測t3", "予測t4"])

# === 道路5 フェーズ時間ログ（★追加） ===
phase_log_path_road5 = r"C:\Users\Tsukasa\Desktop\研究\制御\csv掃き出し\信号J_フェーズ時間ログ_道路5.csv"
phase_log_file_road5 = open(phase_log_path_road5, "w", newline="", encoding="utf-8-sig")
phase_writer_road5   = csv.writer(phase_log_file_road5)
phase_writer_road5.writerow(["回数", "step", "パターンA(秒)", "パターンB(秒)", "予測t3", "予測t4"])

# === 無駄青ログ ===
wasted_green_log_path = r"C:\Users\Tsukasa\Desktop\研究\制御\csv掃き出し\無駄青時間ログ.csv"
wasted_green_file     = open(wasted_green_log_path, "w", newline="", encoding="utf-8-sig")
wasted_green_writer   = csv.writer(wasted_green_file)
wasted_green_writer.writerow(["回数", "step", "監視秒数", "無駄青時間(s)"])

# === 遅れ時間ログ（道路1 主道路南） ===
delay_log_path = r"C:\Users\Tsukasa\Desktop\研究\制御\csv掃き出し\遅れ時間\信号機A_遅れ時間ログ_主道路(南).csv"
delay_log_file = open(delay_log_path, "w", newline="", encoding="utf-8-sig")
delay_writer   = csv.writer(delay_log_file)
delay_writer.writerow(["回数", "step", "前半台数", "後半台数", "delta", "okure", "遅れ時間(秒)"])

noth_delay_log_path = r"C:\Users\Tsukasa\Desktop\研究\制御\csv掃き出し\遅れ時間\信号機A_遅れ時間ログ_主道路(北).csv"
noth_delay_log_file = open(noth_delay_log_path, "w", newline="", encoding="utf-8-sig")
noth_delay_writer   = csv.writer(noth_delay_log_file)
noth_delay_writer.writerow(["回数", "step", "前半台数", "後半台数", "delta", "okure", "遅れ時間(秒)"])

delay_minor_log_path = r"C:\Users\Tsukasa\Desktop\研究\制御\csv掃き出し\遅れ時間\信号機A_遅れ時間_従道路.csv"
delay_minor_log_file = open(delay_minor_log_path, "w", newline="", encoding="utf-8-sig")
delay_minor_writer   = csv.writer(delay_minor_log_file)
delay_minor_writer.writerow(["回数", "step", "道路ID", "前半台数", "後半台数", "delta", "okure", "遅れ時間(秒)"])

# === 道路5 遅れ時間ログ（★追加） ===
delay_log_path_road5 = r"C:\Users\Tsukasa\Desktop\研究\制御\csv掃き出し\遅れ時間\信号機J_遅れ時間ログ_道路5.csv"
delay_log_file_road5 = open(delay_log_path_road5, "w", newline="", encoding="utf-8-sig")
delay_writer_road5   = csv.writer(delay_log_file_road5)
delay_writer_road5.writerow(["回数", "step", "前半台数", "後半台数", "delta", "okure", "遅れ時間(秒)"])

# === 全道路遅れ時間ログ ===
all_delay_log_path = r"C:\Users\Tsukasa\Desktop\研究\制御\csv掃き出し\遅れ時間\全遅れ時間ログ_主道路.csv"
all_delay_log_file = open(all_delay_log_path, "w", newline="", encoding="utf-8-sig")
all_delay_writer   = csv.writer(all_delay_log_file)
all_delay_writer.writerow(["回数", "step", "道路ID", "前半台数", "後半台数", "okure", "遅れ時間(秒)"])

all_delay_minor_log_path = r"C:\Users\Tsukasa\Desktop\研究\制御\csv掃き出し\遅れ時間\全遅れ時間_従道路.csv"
all_delay_minor_log_file = open(all_delay_minor_log_path, "w", newline="", encoding="utf-8-sig")
all_delay_minor_writer   = csv.writer(all_delay_minor_log_file)
all_delay_minor_writer.writerow(["回数", "step", "道路ID", "前半台数", "後半台数", "okure", "遅れ時間(秒)"])

green_log_path = r"C:\Users\Tsukasa\Desktop\研究\制御\csv掃き出し\green_durations.csv"
green_log_file = open(green_log_path, "w", newline="", encoding="utf-8-sig")
green_writer   = csv.writer(green_log_file)
green_writer.writerow(["回数", "step", "種類", "青時間(秒)"])

queue_delay_log_path = r"C:\Users\Tsukasa\Desktop\研究\制御\csv掃き出し\queue_and_delay.csv"
queue_delay_file     = open(queue_delay_log_path, "w", newline="", encoding="utf-8-sig")
queue_delay_writer   = csv.writer(queue_delay_file)
queue_delay_writer.writerow(["回数", "step", "道路ID", "待ち台数", "合計遅れ時間(秒)"])


# ============================================================
# 初期フェーズ時間（固定）
# ============================================================
INITIAL_PHASE_DURATIONS = {
    0: 60,   # 道路1（主道路）フェーズ0
    6: 26    # 道路5（従道路）フェーズ6
}


# ============================================================
# 信号制御関数（道路1）奇数/偶数サイクル対応
# ============================================================
def adjust_signal_A_based_on_prediction(t3, t4, step, prediction_count, pending_counts):
    """
    道路1の予測結果に基づき、J交差点のフェーズ0（主道路青）を調整する。
    奇数サイクル：予測値に基づく制御
    偶数サイクル：前サイクルの制御を打ち消す
    制御確定後、道路5・6の既存スケジュールをapplied_delta_road1で再計算して更新する。
    """
    try:
        global cycle_count_road1, last_phase_delta, signal_A_adjusted_once, applied_delta_road1

        light_id = "J"
        logic = traci.trafficlight.getAllProgramLogics(light_id)[0]
        old_phases = logic.getPhases()

        if len(old_phases) != 12:
            print(f"⚠️ 信号機Jのフェーズ数が想定と異なります: {len(old_phases)} フェーズ")
            return

        # サイクル更新
        cycle_count_road1 += 1

        if cycle_count_road1 % 2 == 1:
            # ===== 奇数サイクル：予測値に基づく制御 =====
            if t3 > t4:
                delta = t3 * 2
            elif t3 < t4:
                delta = -t4 * 2
            else:
                delta = 0
            delta = max(-10, min(delta, 10))
            last_phase_delta = delta  # 次（偶数）サイクル用に保存
            print(f"✅ [道路1制御・奇数#{cycle_count_road1}] delta={delta} (t3={t3}, t4={t4})")

        else:
            # ===== 偶数サイクル：前回の打ち消し =====
            delta = -last_phase_delta
            print(f"✅ [道路1制御・偶数#{cycle_count_road1}] delta={delta} (打ち消し)")

        applied_delta_road1 = delta  # 実際に適用した値を保存（計測タイミング補正用）
        signal_A_adjusted_once = True

        from traci._trafficlight import Phase

        new_phases = []
        for i, phase in enumerate(old_phases):
            if i == 0:
                # フェーズ0（主道路青）：制御対象
                dur = INITIAL_PHASE_DURATIONS[0] + delta
                dur = max(5, dur)
            else:
                # フェーズ6（従道路青）を含む他フェーズ：現在値を保持（道路5制御の結果を壊さない）
                dur = phase.duration
            min_dur = getattr(phase, "minDur", dur)
            max_dur = getattr(phase, "maxDur", dur)
            new_phases.append(Phase(duration=dur, state=phase.state, minDur=min_dur, maxDur=max_dur))

        logic.phases = new_phases
        traci.trafficlight.setProgramLogic(light_id, logic)

        print(f"   → フェーズ0={new_phases[0].duration}s (step={step}, 予測回数={prediction_count})")

        if "phase_writer" in globals():
            phase_writer.writerow([
                prediction_count,
                step,
                new_phases[0].duration,
                new_phases[6].duration,
                t3,
                t4
            ])

        # ============================================================
        # 道路5・6の既存スケジュールを最新の applied_delta_road1 で再計算・更新
        # フェーズ0が確定したので、従道路の赤時間が確定した
        # ============================================================
        updated_keys = set()
        for key, records in pending_counts.items():
            new_records = []
            updated = False
            for record in records:
                if len(record) == 6:
                    record_time, tag, road_id, edges, expected_diff, red_start = record
                else:
                    new_records.append(record)
                    continue

                if road_id in ["5", "6"]:
                    # red_start から最新の applied_delta_road1 で再計算
                    new_t1, new_t2 = get_measurement_times_for_minor_road(
                        red_start, base_t1=44, base_t2=44, delta_adjustment=applied_delta_road1)
                    new_expected_diff = new_t2 - new_t1
                    new_record_time = new_t1 if tag == "t1" else new_t2
                    new_records.append((new_record_time, tag, road_id, edges, new_expected_diff, red_start))
                    updated = True
                else:
                    new_records.append(record)

            pending_counts[key] = new_records
            if updated:
                updated_keys.add(key)

        if updated_keys:
            # 更新後のスケジュールをログ出力
            for key in updated_keys:
                for record in pending_counts[key]:
                    record_time, tag, road_id, edges, expected_diff, red_start = record
                    if tag == "t1":
                        t1_new = record_time
                    else:
                        t2_new = record_time
                # t1/t2両方揃ったタイミングでまとめてログ
                records_56 = [r for r in pending_counts[key] if r[2] in ["5", "6"]]
                if len(records_56) >= 2:
                    t1_log = records_56[0][0]
                    t2_log = records_56[1][0]
                    rid_log = records_56[0][2]
                    rs_log  = records_56[0][5]
                    print(f"🔄 [スケジュール更新] 道路{rid_log} | 赤開始=step{rs_log} | applied_delta1={applied_delta_road1} | t1=step{t1_log} | t2=step{t2_log} | t2-t1={t2_log - t1_log}s")

    except Exception as e:
        print(f"❌ 道路1 信号制御エラー: {e}")


# ============================================================
# 信号制御関数（道路5）奇数/偶数サイクル対応
# ============================================================
def adjust_signal_based_on_prediction_road5(t3, t4, step, prediction_count_road5, pending_counts):
    """
    道路5の予測結果に基づき、J交差点のフェーズ6（従道路青）を調整する。
    奇数サイクル：予測値に基づく制御
    偶数サイクル：前サイクルの制御を打ち消す
    制御確定後、道路1・10の既存スケジュールをapplied_delta_road5で再計算して更新する。
    """
    try:
        global cycle_count_road5, last_phase_delta_road5, signal_road5_adjusted_once, applied_delta_road5

        light_id = "J"
        logic = traci.trafficlight.getAllProgramLogics(light_id)[0]
        old_phases = logic.getPhases()

        if len(old_phases) != 12:
            print(f"⚠️ 信号機Jのフェーズ数が想定と異なります: {len(old_phases)} フェーズ")
            return

        # サイクル更新
        cycle_count_road5 += 1

        if cycle_count_road5 % 2 == 1:
            # ===== 奇数サイクル：予測値に基づく制御 =====
            if t3 > t4:
                delta5 = t3 * 2
            elif t3 < t4:
                delta5 = -t4 * 2
            else:
                delta5 = 0
            delta5 = max(-10, min(delta5, 10))
            last_phase_delta_road5 = delta5  # 次（偶数）サイクル用に保存
            print(f"✅ [道路5制御・奇数#{cycle_count_road5}] delta5={delta5} (t3={t3}, t4={t4})")

        else:
            # ===== 偶数サイクル：前回の打ち消し =====
            delta5 = -last_phase_delta_road5
            print(f"✅ [道路5制御・偶数#{cycle_count_road5}] delta5={delta5} (打ち消し)")

        applied_delta_road5 = delta5  # 実際に適用した値を保存（計測タイミング補正用）
        signal_road5_adjusted_once = True

        from traci._trafficlight import Phase

        new_phases = []
        for i, phase in enumerate(old_phases):
            if i == 6:
                # フェーズ6（従道路青）：制御対象
                dur = INITIAL_PHASE_DURATIONS[6] + delta5
                dur = max(5, dur)
            else:
                # フェーズ0を含む他のフェーズ：道路5制御では変更しない
                dur = phase.duration
            min_dur = getattr(phase, "minDur", dur)
            max_dur = getattr(phase, "maxDur", dur)
            new_phases.append(Phase(duration=dur, state=phase.state, minDur=min_dur, maxDur=max_dur))

        logic.phases = new_phases
        traci.trafficlight.setProgramLogic(light_id, logic)

        print(f"   → フェーズ6={new_phases[6].duration}s (step={step}, 予測回数={prediction_count_road5})")

        if "phase_writer_road5" in globals():
            phase_writer_road5.writerow([
                prediction_count_road5,
                step,
                new_phases[0].duration,
                new_phases[6].duration,
                t3,
                t4
            ])

        # ============================================================
        # 道路1・10の既存スケジュールを最新の applied_delta_road5 で再計算・更新
        # フェーズ6が確定したので、主道路の赤時間が確定した
        # ============================================================
        updated_keys = set()
        for key, records in pending_counts.items():
            new_records = []
            updated = False
            for record in records:
                if len(record) == 6:
                    record_time, tag, road_id, edges, expected_diff, red_start = record
                else:
                    new_records.append(record)
                    continue

                if road_id in ["1", "10"]:
                    # red_start から最新の applied_delta_road5 で再計算
                    new_t1, new_t2 = get_measurement_times_for_main_road(
                        red_start, base_t1=28, base_t2=27, delta_adjustment=applied_delta_road5)
                    new_expected_diff = new_t2 - new_t1
                    new_record_time = new_t1 if tag == "t1" else new_t2
                    new_records.append((new_record_time, tag, road_id, edges, new_expected_diff, red_start))
                    updated = True
                else:
                    new_records.append(record)

            pending_counts[key] = new_records
            if updated:
                updated_keys.add(key)

        if updated_keys:
            for key in updated_keys:
                records_110 = [r for r in pending_counts[key] if r[2] in ["1", "10"]]
                if len(records_110) >= 2:
                    t1_log = records_110[0][0]
                    t2_log = records_110[1][0]
                    rid_log = records_110[0][2]
                    rs_log  = records_110[0][5]
                    print(f"🔄 [スケジュール更新] 道路{rid_log} | 赤開始=step{rs_log} | applied_delta5={applied_delta_road5} | t1=step{t1_log} | t2=step{t2_log} | t2-t1={t2_log - t1_log}s")

    except Exception as e:
        print(f"❌ 道路5 信号制御エラー: {e}")


# ============================================================
# 測定タイミング算出
# ============================================================
def get_measurement_times_for_main_road(sim_time, base_t1, base_t2, delta_adjustment=0):
    """
    道路1（主道路）の赤信号 前半/後半 測定時刻を返す。
    delta_adjustment = last_phase_delta（フェーズ0 の変化量）を渡す。
    フェーズ0が延長（delta>0）されると主道路の赤が短くなるため、
    測定タイミングをその分だけ前倒しする。
    """
    t1_time = int(sim_time + base_t1 + delta_adjustment // 2)
    t2_time = int(sim_time + base_t1 + base_t2 + delta_adjustment)
    return t1_time, t2_time


def get_measurement_times_for_minor_road(sim_time, base_t1, base_t2, delta_adjustment=0):
    """
    道路5/6（従道路）の赤信号 前半/後半 測定時刻を返す。
    delta_adjustment = last_phase_delta_road5（フェーズ6 の変化量）を渡す。
    フェーズ6が延長（delta>0）されると従道路の赤が短くなるため、
    測定タイミングをその分だけ前倒しする。
    """
    t1_time = int(sim_time + base_t1 + delta_adjustment // 2)
    t2_time = int(sim_time + base_t1 + base_t2 + delta_adjustment)
    return t1_time, t2_time


# ==== 時間距離図 =============================================================
import matplotlib.pyplot as plt

# パラメータ
MEASURE_START = 30050  # 計測開始秒
MEASURE_END   = 31000  # 計測終了秒
MAX_MEASURE_VEHICLES = None  # Noneなら無制限、数値なら台数制限

DEFAULT_INTERVAL = 120
DEFAULT_MAX_N = 5

# ================================================================
# ★ ここだけ設定すればOK
# ================================================================

# 使用する .sumocfg または .net.xml のパス
NET_FILE_PATH = r"C:\Users\Tsukasa\Desktop\研究\予測\町モデルデータ\toyama_shouwa.sumocfg"

# 時間距離図に表示する信号交差点（南端→北端の順）
# (TL信号ID, 表示名 [, 赤フェーズmin, 赤フェーズmax])
# 赤フェーズを省略すると RED_PHASE_DEFAULT が使われる
TL_ORDER = [
    ("P", "P交差点"),
    ("O", "O交差点"),
    ("N", "N交差点"),
    ("M", "M交差点"),
    ("J", "J交差点", 2, 11),   # J交差点は独自フェーズ範囲
    ("F", "F交差点"),
    ("E", "E交差点"),
    ("D", "D交差点"),
    ("A", "A交差点", 2, 11),   # A交差点は独自フェーズ範囲
]

# TL_ORDER の代表TL以外に同じ交差点で監視する追加TL信号
# {表示名: [(TL_ID [, 赤フェーズmin, 赤フェーズmax]), ...]}
# 赤フェーズを省略すると RED_PHASE_DEFAULT が使われる
# いずれかの TL が赤なら「その交差点が赤」と判定する（OR条件）
EXTRA_RED_MONITORS = {}

# 赤信号と判定するデフォルトのフェーズ番号の範囲（min以上max以下）
RED_PHASE_DEFAULT = (2, 5)

# ================================================================
# net.xml 自動取得ロジック（変更不要）
# ================================================================

def _resolve_net_file(path):
    """sumocfg の場合はアクティブな net-file を取得して絶対パスを返す"""
    if path.endswith('.sumocfg'):
        base_dir = os.path.dirname(os.path.abspath(path))
        tree = ET.parse(path)
        root = tree.getroot()
        for net_elem in root.iter('net-file'):
            val = net_elem.get('value', '').strip()
            if val:
                return os.path.join(base_dir, val)
        raise FileNotFoundError(f"sumocfg に有効な net-file が見つかりません: {path}")
    return os.path.abspath(path)


def _build_tl_junction_map(net_xml_path):
    """
    connection 要素の via 属性から TL ID → junction node ID のマッピングを構築する。
    via 属性の形式は ':junctionID_接続番号_レーン番号'。
    """
    tree = ET.parse(net_xml_path)
    root = tree.getroot()
    tl_to_junc = {}
    for conn in root.findall('.//connection'):
        tl = conn.get('tl')
        via = conn.get('via')
        if tl and via:
            # ':junctionID_数字_数字' から junctionID を抽出（アンダースコア含む名前にも対応）
            m = re.match(r':(.+)_\d+_\d+$', via)
            if m:
                tl_to_junc[tl] = m.group(1)
    return tl_to_junc


def _bfs_between_nodes(net, from_node, to_node, max_hops=300):
    """
    from_node から to_node までの最短エッジ列を BFS で探索して返す。
    内部エッジ（交差点内）は sumolib が withInternal=False の場合除外済み。
    """
    from collections import deque

    target_id = to_node.getID()
    visited = {from_node.getID()}
    queue = deque()

    for edge in from_node.getOutgoing():
        nxt = edge.getToNode()
        if nxt.getID() not in visited:
            queue.append((nxt, [edge]))

    while queue:
        node, path = queue.popleft()
        node_id = node.getID()

        if node_id == target_id:
            return path

        if node_id in visited or len(path) >= max_hops:
            continue
        visited.add(node_id)

        for edge in node.getOutgoing():
            nxt_id = edge.getToNode().getID()
            if nxt_id not in visited:
                queue.append((edge.getToNode(), path + [edge]))

    return []


def setup_from_net(net_file_or_sumocfg, tl_order):
    """
    .net.xml を読み込み、指定した信号交差点順にルートを BFS で自動探索して
    エッジリスト・エッジ長・信号交差点位置を返す。

    Parameters
    ----------
    net_file_or_sumocfg : str
        .net.xml または .sumocfg のパス
    tl_order : list of (tl_id, label)
        南端→北端の順で信号交差点を指定。例: [("P","P交差点"), ..., ("A","A交差点")]

    Returns
    -------
    south_edges : list[str]
        南→北方向のエッジ ID 列
    north_edges : list[str]
        北→南方向のエッジ ID 列
    edge_lengths : dict[str, int]
        {エッジID: 長さ[m]}（四捨五入）
    signal_intersections : dict[str, int]
        {交差点名: 南端からの累積距離[m]}
    """
    net_file = _resolve_net_file(net_file_or_sumocfg)
    net = sumolib.net.readNet(net_file, withInternal=False)

    tl_junc_map = _build_tl_junction_map(net_file)

    # TL ID → sumolib Node
    tl_nodes = {}
    for entry in tl_order:
        tl_id, label = entry[0], entry[1]
        junc_id = tl_junc_map.get(tl_id)
        if junc_id is None:
            raise ValueError(f"TL '{tl_id}' に対応する junction が見つかりません（net.xml に tl='{tl_id}' の connection がない可能性があります）")
        try:
            tl_nodes[tl_id] = net.getNode(junc_id)
        except Exception:
            raise ValueError(f"TL '{tl_id}' の junction '{junc_id}' を sumolib で取得できませんでした")

    tl_id_list = [t[0] for t in tl_order]

    # --- 南→北ルート（P → A 方向）---
    south_edges_obj = []
    for i in range(len(tl_id_list) - 1):
        seg = _bfs_between_nodes(net, tl_nodes[tl_id_list[i]], tl_nodes[tl_id_list[i + 1]])
        if not seg:
            raise RuntimeError(
                f"{tl_id_list[i]}→{tl_id_list[i+1]} 間のルートが見つかりません。"
                "BFS の max_hops を増やすか、TL_ORDER を確認してください。"
            )
        south_edges_obj.extend(seg)

    # --- 北→南ルート（A → P 方向）---
    north_edges_obj = []
    north_segments = []  # [(seg_edges, from_tl_id, to_tl_id), ...]
    for i in range(len(tl_id_list) - 1, 0, -1):
        seg = _bfs_between_nodes(net, tl_nodes[tl_id_list[i]], tl_nodes[tl_id_list[i - 1]])
        if not seg:
            raise RuntimeError(
                f"{tl_id_list[i]}→{tl_id_list[i-1]} 間の逆ルートが見つかりません。"
                "BFS の max_hops を増やすか、TL_ORDER を確認してください。"
            )
        north_edges_obj.extend(seg)
        north_segments.append((seg, tl_id_list[i], tl_id_list[i - 1]))

    south_edges = [e.getID() for e in south_edges_obj]
    north_edges = [e.getID() for e in north_edges_obj]

    # エッジ長（最初のレーンの長さを使用）
    edge_lengths = {}
    for e in south_edges_obj + north_edges_obj:
        eid = e.getID()
        if eid not in edge_lengths:
            edge_lengths[eid] = round(e.getLanes()[0].getLength())

    # 各 TL 交差点の累積距離（南端 = 0m）
    junc_id_to_info = {tl_nodes[e[0]].getID(): (e[0], e[1]) for e in tl_order}
    # 起点交差点（南端 = 0m）を先に登録する
    # （起点は FROM ノードのため、エッジ探索ループには現れない）
    signal_intersections = {tl_order[0][1]: 0}
    cum = 0
    for edge in south_edges_obj:
        eid = edge.getID()
        cum += edge_lengths.get(eid, 0)
        to_junc_id = edge.getToNode().getID()
        if to_junc_id in junc_id_to_info:
            _, label = junc_id_to_info[to_junc_id]
            signal_intersections[label] = cum

    # --- 北方向エッジ → 南軸 X 座標マップ ---
    # 各北方向エッジの pos=0（A側）と pos=length（P側）の南軸 x を
    # TL 交差点位置をアンカーとして比例補間で算出する。
    tl_id_to_label = {e[0]: e[1] for e in tl_order}
    north_edge_x_map = {}  # {edge_id: (x_at_pos0, x_at_pos_end)}
    for seg_edges, from_tl, to_tl in north_segments:
        x_from = signal_intersections.get(tl_id_to_label.get(from_tl, ""), None)
        x_to   = signal_intersections.get(tl_id_to_label.get(to_tl,   ""), None)
        if x_from is None or x_to is None:
            continue
        total_seg_len = sum(edge_lengths.get(e.getID(), 0) for e in seg_edges)
        if total_seg_len == 0:
            continue
        cum = 0
        for edge in seg_edges:
            eid  = edge.getID()
            elen = edge_lengths.get(eid, 0)
            # pos=0 は A 側（x_from），pos=elen は P 側（x_to）に向かって線形補間
            x0 = x_from + (cum / total_seg_len) * (x_to - x_from)
            x1 = x_from + ((cum + elen) / total_seg_len) * (x_to - x_from)
            north_edge_x_map[eid] = (x0, x1)
            cum += elen

    print("=== net.xml から自動取得したルート情報 ===")
    print(f"南→北エッジ列: {south_edges}")
    print(f"北→南エッジ列: {north_edges}")
    print("エッジ長:")
    for eid in south_edges:
        print(f"  {eid}: {edge_lengths[eid]} m")
    print("信号交差点位置（南端からの累積距離）:")
    for name, pos in signal_intersections.items():
        print(f"  {name}: {pos} m")

    return south_edges, north_edges, edge_lengths, signal_intersections, north_edge_x_map


# net.xml からルート・距離・交差点位置を自動取得
(SOUTH_EDGES, NORTH_EDGES,
 EDGE_LENGTHS, SIGNAL_INTERSECTIONS,
 NORTH_EDGE_X_MAP) = setup_from_net(NET_FILE_PATH, TL_ORDER)

SOUTH_START_EDGE    = SOUTH_EDGES[0]   # 南方向の計測開始エッジ（P交差点直後）
SOUTH_COMPLETE_EDGE = SOUTH_EDGES[-1]  # 南方向の計測完了エッジ（A交差点通過後）
NORTH_START_EDGE    = NORTH_EDGES[0]   # 北方向の計測開始エッジ（A交差点直後）
NORTH_COMPLETE_EDGE = NORTH_EDGES[-1]  # 北方向の計測完了エッジ（P交差点通過後）

# 記録用
enter_time_south = {}
enter_time_north = {}
measured_south = set()
measured_north = set()

vehicle_traces = defaultdict(list)
completed_south = set()
completed_north = set()

# 信号赤時間記録用
signal_red_intervals = defaultdict(list)   # {交差点名: [[start, end], ...]}
red_active_states = {}                     # {交差点名: bool}


# ===============================
# ログ取得関数
# ===============================
def log_vehicle_positions(sim_time):
    global enter_time_south, enter_time_north
    global measured_south, measured_north
    global vehicle_traces, completed_south, completed_north
    global signal_red_intervals, red_active_states

    if sim_time < MEASURE_START or sim_time > MEASURE_END:
        return

    # === 各交差点の赤時間を監視 ===
    # TL_ORDER と EXTRA_RED_MONITORS を統合し、交差点ごとに一括判定する
    # (OR条件: いずれかの TL が赤なら「その交差点が赤」)
    intersection_red = {}

    # TL_ORDER の代表TLをチェック
    for entry in TL_ORDER:
        tl_id, name = entry[0], entry[1]
        red_min, red_max = entry[2:4] if len(entry) >= 4 else RED_PHASE_DEFAULT
        phase = traci.trafficlight.getPhase(tl_id)
        # まだ赤と判定されていない場合のみ上書き（OR条件）
        if name not in intersection_red:
            intersection_red[name] = False
        if red_min <= phase <= red_max:
            intersection_red[name] = True

    # EXTRA_RED_MONITORS の追加TLをチェック（OR条件で合算）
    for name, monitors in EXTRA_RED_MONITORS.items():
        if name not in intersection_red:
            intersection_red[name] = False
        for mon in monitors:
            tl_id = mon[0]
            red_min, red_max = (mon[1], mon[2]) if len(mon) >= 3 else RED_PHASE_DEFAULT
            phase = traci.trafficlight.getPhase(tl_id)
            if red_min <= phase <= red_max:
                intersection_red[name] = True
                break  # この交差点はもう赤確定

    # red_active_states を更新
    for name, red_now in intersection_red.items():
        if red_now:
            if not red_active_states.get(name, False):
                signal_red_intervals[name].append([sim_time, None])
                red_active_states[name] = True
        else:
            if red_active_states.get(name, False):
                if signal_red_intervals[name]:
                    signal_red_intervals[name][-1][1] = sim_time
                red_active_states[name] = False

    south_edge_set = set(SOUTH_EDGES)
    north_edge_set = set(NORTH_EDGES)

    # === 南北方向の検知・記録 ===
    for veh_id in traci.vehicle.getIDList():
        edge_id = traci.vehicle.getRoadID(veh_id)

        # 南方向: 起点エッジへの進入を検知
        if veh_id not in enter_time_south and edge_id == SOUTH_START_EDGE:
            if (MAX_MEASURE_VEHICLES is None) or (len(measured_south) < MAX_MEASURE_VEHICLES):
                enter_time_south[veh_id] = sim_time
                measured_south.add(veh_id)

        # 北方向: 起点エッジへの進入を検知
        if veh_id not in enter_time_north and edge_id == NORTH_START_EDGE:
            if (MAX_MEASURE_VEHICLES is None) or (len(measured_north) < MAX_MEASURE_VEHICLES):
                enter_time_north[veh_id] = sim_time
                measured_north.add(veh_id)

        # 南方向の記録
        if veh_id in measured_south and edge_id in south_edge_set:
            pos = traci.vehicle.getLanePosition(veh_id)
            vehicle_traces[veh_id].append(("south", sim_time, edge_id, pos))
            if edge_id == SOUTH_COMPLETE_EDGE:
                completed_south.add(veh_id)

        # 北方向の記録
        if veh_id in measured_north and edge_id in north_edge_set:
            pos = traci.vehicle.getLanePosition(veh_id)
            vehicle_traces[veh_id].append(("north", sim_time, edge_id, pos))
            if edge_id == NORTH_COMPLETE_EDGE:
                completed_north.add(veh_id)


# ===============================
# 座標変換
# ===============================
def convert_x_south(row):
    """南→北方向: 南端(0m)からの累積距離に変換"""
    x = 0
    for edge in SOUTH_EDGES:
        if row["edge"] == edge:
            return x + row["pos"]
        x += EDGE_LENGTHS[edge]
    return None

def convert_x_north(row):
    """北→南方向: 南軸（P=0m, A=総距離m）上の x 座標に変換する。
    TL 交差点位置をアンカーとした比例補間マップ NORTH_EDGE_X_MAP を使用する。"""
    params = NORTH_EDGE_X_MAP.get(row["edge"])
    if params is None:
        return None
    x0, x1 = params          # x0: pos=0(A側)の x, x1: pos=length(P側)の x
    elen = EDGE_LENGTHS.get(row["edge"], 0)
    if elen == 0:
        return x0
    return x0 + (row["pos"] / elen) * (x1 - x0)


# ===============================
# プロット関数
# ===============================
def plot_time_space_diagram():
    # === デバッグ診断 ===
    print(f"\n[DEBUG] measured_north の車両数: {len(measured_north)}")
    print(f"[DEBUG] NORTH_START_EDGE: {NORTH_START_EDGE}")
    print(f"[DEBUG] NORTH_EDGES (先頭5): {NORTH_EDGES[:5]}")
    print(f"[DEBUG] NORTH_EDGE_X_MAP エントリ数: {len(NORTH_EDGE_X_MAP)}")
    print(f"[DEBUG] NORTH_EDGE_X_MAP (先頭3): {list(NORTH_EDGE_X_MAP.items())[:3]}")
    for veh_id in list(measured_north)[:3]:
        records = [r for r in vehicle_traces[veh_id] if r[0] == "north"]
        edges_used = list(dict.fromkeys(r[2] for r in records))
        in_map = [e for e in edges_used if e in NORTH_EDGE_X_MAP]
        print(f"[DEBUG] 車両 {veh_id}: north記録={len(records)}件, エッジ={edges_used[:5]}, マップ一致={in_map[:5]}")
    # ==================

    plt.figure(figsize=(14, 7))

    def draw_traces(df, color):
        plt.plot(df["x"], df["sim_time"], color=color, alpha=0.4)

    # 南方向（P→A：A到達完了に限らず、P進入を検知した全車両をプロット）
    for veh_id in measured_south:
        if veh_id not in enter_time_south:
            continue
        records = [r for r in vehicle_traces[veh_id] if r[0] == "south"]
        if not records:
            continue
        df = pd.DataFrame(records, columns=["dir", "sim_time", "edge", "pos"])
        df["x"] = df.apply(convert_x_south, axis=1)
        df = df.dropna(subset=["x"])
        if df.empty:
            continue
        draw_traces(df, "blue")

    # 北方向（A→P：P到達完了に限らず、A進入を検知した全車両をプロット）
    for veh_id in measured_north:
        if veh_id not in enter_time_north:
            continue
        records = [r for r in vehicle_traces[veh_id] if r[0] == "north"]
        if not records:
            continue
        df = pd.DataFrame(records, columns=["dir", "sim_time", "edge", "pos"])
        df["x"] = df.apply(convert_x_north, axis=1)
        df = df.dropna(subset=["x"])
        if df.empty:
            continue
        draw_traces(df, "green")

    # 軸ラベル
    plt.xlabel("リンク距離 [m]")
    plt.ylabel("シミュレーション時間 [s]")
    plt.title("時間距離図（信号制御付き, シミュレーション時間基準）")

    # 主目盛り: 南端 → 各TL交差点 → 北端（TL_ORDER から自動生成）
    tick_positions = [0]
    tick_labels    = [TL_ORDER[0][1]]  # 南端 = P交差点
    for entry in TL_ORDER[1:]:
        label = entry[1]
        if label in SIGNAL_INTERSECTIONS:
            tick_positions.append(SIGNAL_INTERSECTIONS[label])
            tick_labels.append(label)
    total_south = sum(EDGE_LENGTHS[e] for e in SOUTH_EDGES)
    tick_positions.append(total_south)
    tick_labels.append("北端")

    plt.xticks(tick_positions, tick_labels)
    plt.grid(True, which="major", linestyle="--", alpha=0.6)

    # 赤信号時間帯の描画（sim_time 基準）
    for name, x_pos in SIGNAL_INTERSECTIONS.items():
        if name in signal_red_intervals:
            for start, end in signal_red_intervals[name]:
                if end is None:
                    end = MEASURE_END
                plt.fill_betweenx([start, end],
                                  x_pos - 5, x_pos + 5,
                                  color="red", alpha=0.3)

    # 凡例
    legend_lines = [
        plt.Line2D([0], [0], color="blue", lw=2, label="南→北"),
        plt.Line2D([0], [0], color="green", lw=2, label="北→南"),
        plt.Rectangle((0, 0), 1, 1, color="red", alpha=0.3, label="赤信号")
    ]
    plt.legend(handles=legend_lines)

    plt.show()
# ============================================================================
# ============================================================================


# ============================================================
# グローバル状態
# ============================================================
last_phase_delta         = None   # 奇数サイクルで保存した予測ベースのdelta（偶数サイクルの打ち消し用）
applied_delta_road1      = 0      # 直前サイクルで実際に適用したdelta（計測タイミング補正用）
signal_A_adjusted_once   = False
cycle_count_road1        = 0      # 道路1 制御サイクルカウンタ

# ★ 道路5 用グローバル
last_phase_delta_road5   = None   # 奇数サイクルで保存した予測ベースのdelta5（偶数サイクルの打ち消し用）
applied_delta_road5      = 0      # 直前サイクルで実際に適用したdelta5（計測タイミング補正用）
signal_road5_adjusted_once   = False
prediction_count_road5       = 0
cycle_count_road5            = 0   # 道路5 制御サイクルカウンタ
latest_data_road5 = {rid: None for rid in used_road_ids}

green_monitoring_active      = False
green_monitor_start_time     = None
green_monitor_end_time       = None
green_monitor_no_pass_seconds = 0
green_monitor_phase_end_time = None

main_green_durations  = []
minor_green_durations = []


# ============================================================
# シミュレーション本体
# ============================================================
def run_simulation(sumocfg_path, log_dir_path, control_signal=True):
    global prediction_count, last_phase_delta, signal_A_adjusted_once, cycle_count_road1, applied_delta_road1
    global prediction_count_road5, last_phase_delta_road5, signal_road5_adjusted_once, cycle_count_road5, applied_delta_road5
    global latest_data_road5
    global green_monitoring_active, green_monitor_start_time
    global green_monitor_end_time, green_monitor_no_pass_seconds
    global green_monitor_phase_end_time

    if not os.path.exists(log_dir_path):
        os.makedirs(log_dir_path)

    # === スケーラー読み込み ===
    if not os.path.exists(scaler_path_road1):
        raise FileNotFoundError(f"❌ 道路1スケーラーが存在しません: {scaler_path_road1}")
    scaler_road1 = joblib.load(scaler_path_road1)
    print("✅ 道路1スケーラー読み込み完了")

    # ★ 道路5 スケーラー読み込み
    if not os.path.exists(scaler_path_road5):
        raise FileNotFoundError(f"❌ 道路5スケーラーが存在しません: {scaler_path_road5}")
    scaler_road5 = joblib.load(scaler_path_road5)
    print("✅ 道路5スケーラー読み込み完了")

    sumoBinary = sumolib.checkBinary('sumo-gui')
    traci.start([sumoBinary, "-c", sumocfg_path, "--start", "--quit-on-end"])
    sim_time = 0
    prev_states           = defaultdict(dict)
    pending_counts        = defaultdict(list)
    first_half_storage    = defaultdict(dict)

    writers = {}
    scheduled_times_A = set()
    last_phase_delta_before_prediction       = 0
    last_phase_delta_road5_before_prediction = 0   # ★ 道路5 用

    vehicle_delay_time = {}

    queue_delay_writers = {}
    vehicle_delay_time  = {}
    cycle_stopped_vehicles = {rid: set() for rid in ["1", "5", "6", "10"]}

    # === 交差点A 出力フォルダ ===
    a_dir = os.path.join(log_dir_path, "交差点A")
    os.makedirs(a_dir, exist_ok=True)

    total_file_path = os.path.join(a_dir, "交差点A_total_queue_delay.csv")
    total_file      = open(total_file_path, "w", newline="", encoding="utf-8-sig")
    total_writer    = csv.writer(total_file)
    total_writer.writerow(["サイクル", "step", "全道路停車車両数合計", "全道路遅れ時間合計"])

    queue_delay_files   = {}
    queue_delay_writers = {}

    for road_id in ["1", "5", "6", "10"]:
        file_path = os.path.join(a_dir, f"道路{road_id}_queue_delay.csv")
        f      = open(file_path, "w", newline="", encoding="utf-8-sig")
        writer = csv.writer(f)
        writer.writerow(["サイクル", "step", "道路ID", "待ち台数合計", "遅れ時間合計"])
        queue_delay_files[road_id]   = f
        queue_delay_writers[road_id] = writer

    cycle_queue_counts = {rid: [] for rid in ["1", "12", "10", "11"]}
    cycle_delay_sums   = {rid: [] for rid in ["1", "12", "10", "11"]}
    cycle_counter      = 0
    last_phase_A       = None

    main_green_durations  = []
    minor_green_durations = []
    green_start_time      = None
    current_green_type    = None

    cycle_count = 0
    green_state = {"start": None, "type": None}

    tls_id = "A"
    MAIN_GREEN_PHASES  = [0]
    MINOR_GREEN_PHASES = [6]

    for tl_id, configs in TRAFFIC_LIGHT_CONFIG.items():
        for conf in configs:
            road_id = conf["id"]
            f = open(os.path.join(log_dir_path, f"道路{road_id}.csv"), "w", newline="", encoding="utf-8")
            writer = csv.writer(f)
            writer.writerow(["step", "count"])
            writers[road_id] = writer

    print("🚦 シミュレーション開始...")

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        sim_time = traci.simulation.getTime()

        phase = traci.trafficlight.getPhase("J")

        if phase in MAIN_GREEN_PHASES or phase in MINOR_GREEN_PHASES:
            if green_state["start"] is None:
                green_state["start"] = sim_time
                green_state["type"]  = "主道路" if phase in MAIN_GREEN_PHASES else "従道路"
        else:
            if green_state["start"] is not None:
                duration = sim_time - green_state["start"]
                cycle_count += 1
                green_writer.writerow([cycle_count, sim_time, green_state["type"], duration])
                print(f"🟢 {green_state['type']} 青 {duration} 秒 (step={sim_time})")
                green_state["start"] = None
                green_state["type"]  = None

        # === サイクル判定 ===
        current_phase_A = traci.trafficlight.getPhase("J")
        total_phases_A  = len(traci.trafficlight.getAllProgramLogics("J")[0].getPhases())

        if last_phase_A == total_phases_A - 1 and current_phase_A == 0 and sim_time >= 1200:
            cycle_counter += 1
            total_stopped = 0
            total_delay   = 0

            for road_id in ["1", "5", "6", "10"]:
                stopped_count = len(cycle_stopped_vehicles[road_id])
                delay_sum     = sum(vehicle_delay_time.get(veh, 0) for veh in cycle_stopped_vehicles[road_id])

                queue_delay_writers[road_id].writerow(
                    [cycle_counter, sim_time, road_id, stopped_count, delay_sum]
                )
                total_stopped += stopped_count
                total_delay   += delay_sum

            total_writer.writerow([cycle_counter, sim_time, total_stopped, total_delay])
            cycle_stopped_vehicles = {rid: set() for rid in ["1", "5", "6", "10"]}

        last_phase_A = current_phase_A

        # === 時間距離図 ===
        log_vehicle_positions(sim_time)

        # === 待ち台数・遅れ時間の記録 ===
        if sim_time >= 1200:
            for road_id in ["1", "5", "6", "10"]:
                edges = None
                for tl_id, configs in TRAFFIC_LIGHT_CONFIG.items():
                    for conf in configs:
                        if conf["id"] == road_id:
                            edges = conf["edges"]
                            break
                    if edges:
                        break
                if not edges:
                    continue

                for edge in edges:
                    for veh_id in traci.edge.getLastStepVehicleIDs(edge):
                        if traci.vehicle.getSpeed(veh_id) <= 0:
                            cycle_stopped_vehicles[road_id].add(veh_id)
                            if veh_id not in vehicle_delay_time:
                                vehicle_delay_time[veh_id] = 0
                            vehicle_delay_time[veh_id] += 1

        # === 無駄青監視 ===
        if green_monitoring_active and green_monitor_start_time <= sim_time < green_monitor_end_time:
            try:
                passing = traci.edge.getLastStepVehicleIDs("E24")
                if len(passing) == 0:
                    green_monitor_no_pass_seconds += 1
            except Exception as e:
                print(f"⚠️ 無駄青監視エラー: {e}")

        elif green_monitoring_active and sim_time >= green_monitor_end_time:
            monitoring_duration = green_monitor_end_time - green_monitor_start_time
            if prediction_count >= MAX_RECORD_COUNT:
                print("🛑 無駄青記録上限に到達。シミュレーション終了。")
                traci.close()
                _close_all_logs()
                return

            wasted_green_writer.writerow([
                prediction_count,
                green_monitor_phase_end_time,
                green_monitor_end_time - green_monitor_start_time,
                green_monitor_no_pass_seconds
            ])
            green_monitoring_active = False

        # ============================================================
        # 信号状態監視・計測スケジュール
        # ============================================================
        for tl_id, configs in TRAFFIC_LIGHT_CONFIG.items():
            for i, config in enumerate(configs):
                try:
                    key     = (tl_id, i)
                    current = traci.trafficlight.getRedYellowGreenState(tl_id)
                    prev    = prev_states[tl_id].get(i, "")

                    if sim_time >= PREP_TIME and prev == config["red"] and current == config["green"]:
                        road_id = config["id"]
                        edges   = config["edges"]

                        # 無駄青監視
                        if config["edges"] == ["DtoA"]:
                            logic            = traci.trafficlight.getAllProgramLogics("J")[0]
                            phase_0_duration = logic.getPhases()[0].duration
                            green_monitor_start_time     = int(sim_time)
                            green_monitor_end_time       = int(sim_time + phase_0_duration)
                            green_monitor_phase_end_time = green_monitor_end_time
                            green_monitor_no_pass_seconds = 0
                            green_monitoring_active      = True

                        # --- 測定タイミング設定 ---
                        if road_id in ["1", "10"]:
                            # 道路1の赤時間はフェーズ6（従道路青）の実際の適用delta分だけ変化する
                            # → 奇数/偶数問わず実際に適用されたdelta5を使用
                            red_start = int(sim_time)
                            t1, t2 = get_measurement_times_for_main_road(
                                sim_time, base_t1=28, base_t2=28, delta_adjustment=applied_delta_road5)
                            expected_diff = t2 - t1
                            # レコードに red_start を追加（後から再計算するため）
                            pending_counts[key] = [
                                (t1, "t1", road_id, edges, expected_diff, red_start),
                                (t2, "t2", road_id, edges, expected_diff, red_start)
                            ]
                            print(f"📅 [計測スケジュール] 道路{road_id} | 赤開始=step{red_start} | applied_delta5={applied_delta_road5} | t1=step{t1} | t2=step{t2} | t2-t1={expected_diff}s")

                        elif road_id in ["5", "6"]:
                            # 道路5の赤時間はフェーズ0（主道路青）の実際の適用delta分だけ変化する
                            # → 奇数/偶数問わず実際に適用されたdelta1を使用
                            red_start = int(sim_time)
                            t1, t2 = get_measurement_times_for_minor_road(
                                sim_time, base_t1=45, base_t2=45, delta_adjustment=applied_delta_road1)
                            expected_diff = t2 - t1
                            # レコードに red_start を追加（後から再計算するため）
                            pending_counts[key].append((t1, "t1", road_id, edges, expected_diff, red_start))
                            pending_counts[key].append((t2, "t2", road_id, edges, expected_diff, red_start))
                            print(f"📅 [計測スケジュール] 道路{road_id} | 赤開始=step{red_start} | applied_delta1={applied_delta_road1} | t1=step{t1} | t2=step{t2} | t2-t1={expected_diff}s")

                        elif road_id in ["11", "14"]:
                            t1 = int(sim_time + 27); t2 = int(sim_time + 54)
                            pending_counts[key].append((t1, "t1", road_id, edges, 27))
                            pending_counts[key].append((t2, "t2", road_id, edges, 27))

                        elif road_id in ["12", "13"]:
                            t1 = int(sim_time + 44); t2 = int(sim_time + 88)
                            pending_counts[key].append((t1, "t1", road_id, edges, 44))
                            pending_counts[key].append((t2, "t2", road_id, edges, 44))

                        elif road_id in ["15", "18", "19", "22", "24", "25", "27", "29", "30",
                                         "32", "35", "36", "39", "40", "42", "43", "45", "48",
                                         "49", "50", "52", "53", "56", "57", "60", "61"]:
                            t1 = int(sim_time + 42); t2 = int(sim_time + 83)
                            pending_counts[key].append((t1, "t1", road_id, edges, 41))
                            pending_counts[key].append((t2, "t2", road_id, edges, 41))

                        else:
                            t1 = int(sim_time + 21); t2 = int(sim_time + 41)
                            pending_counts[key].append((t1, "t1", road_id, edges, 20))
                            pending_counts[key].append((t2, "t2", road_id, edges, 20))

                    # --- 計測実行 ---
                    new_pending = []
                    for record in pending_counts[key]:
                        # red_start を持つレコード（道路1/5/6/10）と持たないレコードの両方に対応
                        if len(record) == 6:
                            record_time, tag, road_id, edges, expected_diff, red_start = record
                        else:
                            record_time, tag, road_id, edges, expected_diff = record
                            red_start = None
                        if sim_time >= record_time:
                            count = 0
                            for edge in edges:
                                for veh_id in traci.edge.getLastStepVehicleIDs(edge):
                                    if traci.vehicle.getSpeed(veh_id) <= 0:
                                        count += 1

                            try:
                                id_int = int(road_id)
                                if id_int in [1, 2, 3, 4, 9, 12, 23, 25, 28, 40, 47, 50, 51, 54, 55, 58, 59, 62]:
                                    count = math.ceil(count / 2)
                                elif id_int in [5, 6, 7, 8, 10, 11, 14, 15, 18, 19, 30, 31, 34, 35, 37, 38, 42, 44, 45]:
                                    count = math.ceil(count / 2)
                            except:
                                pass

                            if tag == "t1":
                                first_half_storage[road_id][record_time] = (count, expected_diff)
                                writers[road_id].writerow([record_time, count])

                            elif tag == "t2":
                                matched = False
                                for t1_time, (prev_count, diff_exp) in list(first_half_storage[road_id].items()):
                                    if abs(record_time - t1_time - diff_exp) <= 1:
                                        diff = count - prev_count

                                        if diff < 0:
                                            print(f"⚠️ 計測異常: 道路{road_id} の差分が負です。t1={prev_count}, t2={count} → 差={diff} (step={record_time})")

                                        # --- 遅れ時間ログ（既存） ---
                                        if road_id == "6":
                                            # 従道路の赤時間はフェーズ0（主道路青）の延長分だけ長くなる
                                            # → 道路1制御による delta1 を使用
                                            delta_for_log = last_phase_delta_before_prediction if (control_signal and signal_A_adjusted_once) else 0
                                            okure = 91 + delta_for_log
                                            delay_seconds = prev_count * (okure * 0.75) + diff * (okure * 0.25)
                                            delay_minor_writer.writerow([
                                                prediction_count, record_time, road_id,
                                                prev_count, diff, delta_for_log, okure, int(round(delay_seconds))
                                            ])

                                        if road_id == "10":
                                            # 主道路の赤時間はフェーズ6（従道路青）の延長分だけ長くなる
                                            # → 道路5制御による delta5 を使用
                                            delta_for_log = last_phase_delta_road5_before_prediction if (control_signal and signal_road5_adjusted_once) else 0
                                            okure = 57 - delta_for_log
                                            delay_seconds = prev_count * (okure * 0.75) + diff * (okure * 0.25)
                                            noth_delay_writer.writerow([
                                                prediction_count, record_time,
                                                prev_count, diff, delta_for_log, okure, int(round(delay_seconds))
                                            ])

                                        if road_id in ["11", "14"]:
                                            okure = 57
                                            delay_seconds = prev_count * (okure * 0.75) + diff * (okure * 0.25)
                                            all_delay_writer.writerow([
                                                prediction_count, record_time, road_id,
                                                prev_count, diff, okure, int(round(delay_seconds))
                                            ])

                                        if road_id in ["12", "13"]:
                                            okure = 91
                                            delay_seconds = prev_count * (okure * 0.75) + diff * (okure * 0.25)
                                            all_delay_writer.writerow([
                                                prediction_count, record_time, road_id,
                                                prev_count, diff, okure, int(round(delay_seconds))
                                            ])

                                        if road_id in ["11", "14", "16", "17", "20", "21", "23", "26", "2", "28",
                                                        "3", "31", "33", "34", "37", "38", "41", "4", "44", "7",
                                                        "46", "47", "8", "51", "9", "54", "55", "58", "59", "62"]:
                                            okure = 42
                                            delay_seconds = prev_count * (okure * 0.75) + diff * (okure * 0.25)
                                            all_delay_writer.writerow([
                                                prediction_count, record_time, road_id,
                                                prev_count, diff, okure, int(round(delay_seconds))
                                            ])

                                        if road_id in ["15", "18", "19", "22", "24", "25", "27", "29", "30",
                                                        "32", "35", "36", "39", "40", "42", "43", "45", "48",
                                                        "49", "50", "52", "53", "56", "57", "60", "61"]:
                                            okure = 84
                                            delay_seconds = prev_count * (okure * 0.75) + diff * (okure * 0.25)
                                            all_delay_minor_writer.writerow([
                                                prediction_count, record_time, road_id,
                                                prev_count, diff, okure, int(round(delay_seconds))
                                            ])

                                        writers[road_id].writerow([record_time, diff])
                                        del first_half_storage[road_id][t1_time]
                                        latest_data[road_id] = (prev_count, diff)

                                        # ★ 道路5 予測用にもデータを蓄積
                                        latest_data_road5[road_id] = (prev_count, diff)

                                        logger.log_measurement_data(road_id, prev_count, diff, record_time)

                                        # ==========================================
                                        # 道路1 予測・制御
                                        # ==========================================
                                        if road_id == "1":
                                            last_phase_delta_before_prediction = last_phase_delta if last_phase_delta is not None else 0

                                            if all(latest_data[rid] is not None for rid in used_road_ids):
                                                logger.log_prediction_trigger(latest_data, label="[道路1] ")
                                                t1s = [latest_data[rid][0] for rid in used_road_ids]
                                                dts = [latest_data[rid][1] for rid in used_road_ids]

                                                pred = run_prediction(
                                                    t1_array=t1s,
                                                    delay_array=dts,
                                                    edge_index=edge_index,
                                                    scaler=scaler_road1,
                                                    model_path=model_path_road1,
                                                    prediction_count=prediction_count
                                                )

                                                pred_t3, pred_t4 = map(int, np.round(pred))
                                                t1_count = latest_data["1"][0]
                                                diff_val = latest_data["1"][1]

                                                prediction_count += 1
                                                logger.log_prediction_result(
                                                    prediction_count, t1_count, diff_val,
                                                    pred_t3, pred_t4, record_time, label="道路1"
                                                )

                                                if prediction_count >= MAX_RECORD_COUNT:
                                                    print("🛑 道路1 記録上限に到達。シミュレーション終了。")
                                                    traci.close()
                                                    _close_all_logs(queue_delay_files, total_file)
                                                    return

                                                prediction_writer.writerow([
                                                    prediction_count, record_time, t1_count, diff_val, pred_t3, pred_t4
                                                ])

                                                # 道路1の赤時間はフェーズ6（従道路青）の延長分だけ長くなる
                                                # → 道路5制御による delta5 を使用
                                                delta_for_log = last_phase_delta_road5_before_prediction if (control_signal and signal_road5_adjusted_once) else 0
                                                okure = 57 + delta_for_log
                                                delay_seconds = t1_count * (okure * 0.75) + diff_val * (okure * 0.25)
                                                delay_writer.writerow([
                                                    prediction_count, record_time, t1_count, diff_val,
                                                    delta_for_log,
                                                    okure, int(round(delay_seconds))
                                                ])

                                                if control_signal:
                                                    adjust_signal_A_based_on_prediction(
                                                        pred_t3, pred_t4, record_time, prediction_count,
                                                        pending_counts
                                                    )

                                                # latest_data["1"] を次サイクル予測用に更新
                                                t3       = pred_t3
                                                t4_actual    = count
                                                t4_minus_t3  = t4_actual - t3
                                                latest_data["1"] = (t3, t4_minus_t3)

                                            else:
                                                logger.log_data_incomplete(record_time, label="道路1 ")
                                                latest_data["1"] = None

                                        # ==========================================
                                        # ★ 道路5 予測・制御
                                        # ==========================================
                                        if road_id == "5":
                                            last_phase_delta_road5_before_prediction = (
                                                last_phase_delta_road5 if last_phase_delta_road5 is not None else 0
                                            )

                                            if all(latest_data_road5[rid] is not None for rid in used_road_ids):
                                                logger.log_prediction_trigger(latest_data_road5, label="[道路5] ")
                                                t1s_5 = [latest_data_road5[rid][0] for rid in used_road_ids]
                                                dts_5 = [latest_data_road5[rid][1] for rid in used_road_ids]

                                                pred5 = run_prediction(
                                                    t1_array=t1s_5,
                                                    delay_array=dts_5,
                                                    edge_index=edge_index,
                                                    scaler=scaler_road5,
                                                    model_path=model_path_road5,
                                                    prediction_count=prediction_count_road5
                                                )

                                                pred_t3_5, pred_t4_5 = map(int, np.round(pred5))
                                                t1_count_5 = latest_data_road5["5"][0]
                                                diff_val_5 = latest_data_road5["5"][1]

                                                prediction_count_road5 += 1
                                                logger.log_prediction_result(
                                                    prediction_count_road5, t1_count_5, diff_val_5,
                                                    pred_t3_5, pred_t4_5, record_time, label="道路5"
                                                )

                                                prediction_writer_road5.writerow([
                                                    prediction_count_road5, record_time,
                                                    t1_count_5, diff_val_5, pred_t3_5, pred_t4_5
                                                ])

                                                # 道路5 遅れ時間ログ
                                                # 道路5の赤時間はフェーズ0（主道路青）の延長分だけ長くなる
                                                # → 道路1制御による delta1 を使用
                                                delta_for_log5 = last_phase_delta_before_prediction if (control_signal and signal_A_adjusted_once) else 0
                                                okure5 = 91 + delta_for_log5
                                                delay_seconds5 = (
                                                    t1_count_5 * (okure5 * 0.75) +
                                                    diff_val_5  * (okure5 * 0.25)
                                                )
                                                delay_writer_road5.writerow([
                                                    prediction_count_road5, record_time,
                                                    t1_count_5, diff_val_5,
                                                    delta_for_log5, okure5, int(round(delay_seconds5))
                                                ])

                                                if control_signal:
                                                    adjust_signal_based_on_prediction_road5(
                                                        pred_t3_5, pred_t4_5,
                                                        record_time, prediction_count_road5,
                                                        pending_counts
                                                    )

                                                # latest_data_road5["5"] を次サイクル用に更新
                                                t3_5        = pred_t3_5
                                                t4_actual_5 = count
                                                t4_minus_5  = t4_actual_5 - t3_5
                                                latest_data_road5["5"] = (t3_5, t4_minus_5)

                                            else:
                                                logger.log_data_incomplete(record_time, label="道路5 ")
                                                latest_data_road5["5"] = None

                                        matched = True
                                        break

                                if not matched:
                                    new_pending.append(record)
                        else:
                            new_pending.append(record)

                    pending_counts[key] = new_pending
                    prev_states[tl_id][i] = current

                except Exception as e:
                    print(f"⚠ Error at {tl_id} ({config['id']}) - {e}")

    traci.close()
    print("✅ シミュレーション完了")
    _close_all_logs(queue_delay_files, total_file)


def _close_all_logs(queue_delay_files=None, total_file=None):
    """全ログファイルを安全に閉じる。"""
    for obj in [
        phase_log_file, phase_log_file_road5,
        wasted_green_file,
        delay_log_file, delay_log_file_road5,
        noth_delay_log_file, delay_minor_log_file,
        all_delay_log_file, all_delay_minor_log_file,
        green_log_file, queue_delay_file,
        prediction_log_file, prediction_log_file_road5,
    ]:
        try:
            obj.close()
        except Exception:
            pass
    if queue_delay_files:
        for f in queue_delay_files.values():
            try:
                f.close()
            except Exception:
                pass
    if total_file:
        try:
            total_file.close()
        except Exception:
            pass


# ============================================================
# ポスト処理・グラフ描画（元のまま）
# ============================================================
import os
import shutil

def postprocess_logs(log_dir_path):
    signal_to_roads = defaultdict(list)
    for signal, configs in TRAFFIC_LIGHT_CONFIG.items():
        for conf in configs:
            signal_to_roads[signal].append(conf["id"])

    total_counts_by_signal = {}

    for signal, road_ids in signal_to_roads.items():
        signal_dir = os.path.join(log_dir_path, f"信号{signal}")
        os.makedirs(signal_dir, exist_ok=True)
        combined_df = None

        for road_id in road_ids:
            src_path = os.path.join(log_dir_path, f"道路{road_id}.csv")
            dst_path = os.path.join(signal_dir, f"道路{road_id}.csv")
            if not os.path.exists(src_path):
                continue
            shutil.move(src_path, dst_path)
            df = pd.read_csv(dst_path, encoding="utf-8-sig")
            df = df.rename(columns={"count": f"道路{road_id}"})
            if combined_df is None:
                combined_df = df
            else:
                combined_df = pd.merge(combined_df, df, on="step", how="outer")

        if combined_df is not None:
            combined_df = combined_df.sort_values("step").fillna(0)
            combined_df["合計"] = combined_df.drop(columns="step").sum(axis=1)
            signal_sum = int(combined_df["合計"].sum())
            combined_df.to_csv(os.path.join(signal_dir, f"信号{signal}_合計.csv"), index=False, encoding="utf-8-sig")
            total_counts_by_signal[signal] = signal_sum
        else:
            total_counts_by_signal[signal] = 0

    all_total = sum(total_counts_by_signal.values())
    total_df  = pd.DataFrame([total_counts_by_signal])
    total_df["全体合計"] = all_total
    total_df.to_csv(os.path.join(log_dir_path, "全体合計.csv"), index=False, encoding="utf-8-sig")
    print("✅ ポスト処理完了：信号機別フォルダ生成と全体合計出力")


def plot_phase_duration_log(csv_path):
    import matplotlib
    matplotlib.rcParams['font.family'] = 'Meiryo'
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        print("⚠️ フェーズ時間ログが空または存在しません。")
        return
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if df.empty or "回数" not in df.columns:
        print("⚠️ データがありません or '回数' 列が存在しません。")
        return
    total_rows   = len(df)
    rows_per_hour = total_rows / 24
    tick_labels  = [f"{i:02d}:00" for i in range(25)]
    tick_indices = [int(i * rows_per_hour) for i in range(25)]
    tick_indices = [i for i in tick_indices if i < total_rows]
    tick_labels  = tick_labels[:len(tick_indices)]
    plt.figure(figsize=(12, 5))
    plt.plot(df["回数"], df["パターンA(秒)"], label="パターンA（青）", marker="o")
    plt.xlabel("時刻"); plt.ylabel("フェーズ秒数")
    plt.title("信号J パターンA（青信号）時間の推移（道路1制御）")
    plt.xticks(ticks=tick_indices, labels=tick_labels, rotation=45)
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()


def plot_wasted_green_log(csv_path):
    import matplotlib
    matplotlib.rcParams['font.family'] = 'Meiryo'
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        print("⚠️ 無駄青時間ログが空または存在しません。")
        return
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if df.empty or "回数" not in df.columns:
        print("⚠️ データがありません or '回数' 列が存在しません。")
        return
    total_rows    = len(df)
    rows_per_hour = total_rows / 24
    tick_labels   = [f"{i:02d}:00" for i in range(25)]
    tick_indices  = [int(i * rows_per_hour) for i in range(25)]
    tick_indices  = [i for i in tick_indices if i < total_rows]
    tick_labels   = tick_labels[:len(tick_indices)]
    plt.figure(figsize=(12, 5))
    plt.plot(df["回数"], df["無駄青時間(s)"], marker='o', label="無駄青時間")
    plt.xlabel("時刻（1時間ごと）"); plt.ylabel("無駄青時間（秒）")
    plt.title("信号A 無駄青時間（24:00固定）")
    plt.xticks(ticks=tick_indices, labels=tick_labels, rotation=45)
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()


def plot_combined_phase_and_wasted_log(phase_log_path, wasted_log_path):
    import matplotlib
    from matplotlib.ticker import MaxNLocator
    matplotlib.rcParams['font.family'] = 'Meiryo'
    if not os.path.exists(phase_log_path) or not os.path.exists(wasted_log_path):
        print("⚠️ ログファイルが存在しません。")
        return
    df_phase  = pd.read_csv(phase_log_path,  encoding="utf-8-sig")
    df_wasted = pd.read_csv(wasted_log_path, encoding="utf-8-sig")
    if df_phase.empty or df_wasted.empty:
        print("⚠️ ログファイルの中身が空です。")
        return
    total_rows    = len(df_phase)
    rows_per_hour = total_rows / 24
    tick_labels   = [f"{i:02d}:00" for i in range(25)]
    tick_indices  = [int(i * rows_per_hour) for i in range(25)]
    tick_indices  = [i for i in tick_indices if i < total_rows]
    tick_labels   = tick_labels[:len(tick_indices)]
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.set_xlabel("時刻 (h）"); ax1.set_ylabel("青時間時間（s）", color="tab:blue")
    ax1.plot(df_phase["回数"], df_phase["パターンA(秒)"], color="tab:blue", marker="o", label="フェーズ0")
    ax1.tick_params(axis='y', labelcolor="tab:blue")
    ax1.set_xticks(tick_indices); ax1.set_xticklabels(tick_labels, rotation=45)
    ax1.yaxis.set_major_locator(MaxNLocator(nbins='auto', integer=True))
    ax2 = ax1.twinx()
    ax2.set_ylabel("無駄青時間（s）", color="tab:red")
    ax2.plot(df_wasted["回数"], df_wasted["無駄青時間(s)"], color="tab:red", marker="x", label="無駄青")
    ax2.tick_params(axis='y', labelcolor="tab:red")
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title("青時間と無駄青時間の推移"); fig.tight_layout(); plt.grid(True); plt.show()


def plot_green_duration_log(csv_path):
    import matplotlib
    import datetime
    import numpy as np
    matplotlib.rcParams['font.family'] = 'Meiryo'
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        print("⚠️ 青時間ログが空または存在しません。")
        return
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if "step" not in df.columns or "青時間(秒)" not in df.columns:
        print("⚠️ 必要な列(step, 青時間(秒))が存在しません。")
        return
    plt.figure(figsize=(12, 5))
    df_main  = df[df["種類"] == "主道路"]
    df_minor = df[df["種類"] == "従道路"]
    plt.plot(df_main["step"],  df_main["青時間(秒)"],  marker="o", label="主道路", color="tab:blue")
    plt.plot(df_minor["step"], df_minor["青時間(秒)"], marker="s", label="従道路", color="tab:orange")
    max_step = df["step"].max()
    xticks   = np.arange(0, max_step + 3600, 3600)
    xticklabels = [f"{int(h):02d}:00" for h in range(len(xticks))]
    plt.xticks(xticks, xticklabels, rotation=45)
    plt.xlabel("現実時間 (hh:mm)"); plt.ylabel("青信号時間 [秒]")
    plt.title("主道路・従道路の青信号時間の経時変化")
    plt.grid(True, linestyle="--", alpha=0.7); plt.legend(); plt.tight_layout(); plt.show()


# ============================================================
# エントリーポイント
# ============================================================
if __name__ == "__main__":
    run_simulation(sumocfg_path, log_dir_path, control_signal=True)
    postprocess_logs(log_dir_path)
    plot_phase_duration_log(phase_log_path)
    plot_wasted_green_log(wasted_green_log_path)
    plot_combined_phase_and_wasted_log(phase_log_path, wasted_green_log_path)
    plot_green_duration_log(green_log_path)
    plot_time_space_diagram()