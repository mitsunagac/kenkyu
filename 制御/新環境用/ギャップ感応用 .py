import os
import csv
import math
import torch
import traci
import sumolib
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
import warnings
import joblib
import csv
import os
from collections import defaultdict


class TrafficPredictionGNN(torch.nn.Module):
    def __init__(self, num_node_features, dropout_rate=0.2):
        super(TrafficPredictionGNN, self).__init__()
        from torch_geometric.nn import GCNConv
        self.conv1 = GCNConv(num_node_features, 18)
        self.conv2 = GCNConv(18, 18)
        self.conv3 = GCNConv(18, 24)
        self.lin = torch.nn.Linear(24, 2)

        # self.conv1 = GCNConv(num_node_features, 32)
        # self.conv2 = GCNConv(32, 32)
        # self.conv3 = GCNConv(32, 64)
        # self.conv4 = GCNConv(64, 128)
        # self.conv5 = GCNConv(128, 128)
        # self.lin = torch.nn.Linear(128, 2)

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.relu(self.conv3(x, edge_index))
        # x = torch.relu(self.conv4(x, edge_index))
        # x = torch.relu(self.conv5(x, edge_index))
        out = self.lin(x)
        return torch.relu(out)  # ✅ 負値を出さないようにする

# def add_time_and_continuous_label_to_features(features, time_label, time_step, total_steps_in_day=1330, num_classes=5):
#     one_hot = torch.nn.functional.one_hot(torch.tensor(time_label), num_classes=num_classes).float()
#     one_hot = one_hot.unsqueeze(0).repeat(features.shape[0], 1).to(features.device)
#     continuous_time = torch.tensor([time_step / total_steps_in_day], dtype=torch.float).repeat(features.shape[0], 1).to(features.device)
#     return torch.cat([features, one_hot, continuous_time], dim=1)

shown_warnings = set()

def show_warning_once(message):
    if message not in shown_warnings:
        warnings.warn(message, FutureWarning)
        shown_warnings.add(message)

def get_time_label_and_step(step, total_steps_in_day=1330, num_classes=5, prep_time=1300):
    """
    準備時間（prep_time）より前のstepには時間ラベルを付けない（エラーを出す）関数。
    """
    if step < prep_time:
        raise ValueError(f"❌ step={step} は準備時間 {prep_time} 秒未満です。時間ラベルは付与されません。")

    step_in_day = step % total_steps_in_day
    block_size = total_steps_in_day // num_classes
    time_label = step_in_day // block_size

    return time_label, step_in_day


def run_prediction(t1_array, delay_array, edge_index, scaler, model_path, prediction_count):
    """
    学習時と一致する、時間ラベルなしのリアルタイム予測関数。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === 正規化処理 ===
    x_t = np.array(t1_array).reshape(1, -1)
    x_t1 = np.array(delay_array).reshape(1, -1)
    x_t = scaler.transform(x_t).T  # [ノード数, 1]
    x_t1 = scaler.transform(x_t1).T

    x_t = torch.tensor(x_t, dtype=torch.float).to(device)
    x_t1 = torch.tensor(x_t1, dtype=torch.float).to(device)

    # === 特徴結合（時間ラベルなし） ===
    x_input = torch.cat([x_t, x_t1], dim=1).to(device)  # [ノード数, 2]

    # === モデル読み込みと推論 ===
    model = TrafficPredictionGNN(num_node_features=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        out = model(x_input, edge_index.to(device)).cpu().numpy()  # [ノード数, 2]

    # === 逆正規化 ===
    for i in range(2):
        out[:, i] = scaler.inverse_transform(out[:, i].reshape(1, -1))[0]

    # === 負値チェック（警告表示） ===
    if (out < 0).any():
        print("⚠️ 予測結果にマイナス値があります。修正推奨")
        neg_idx = np.argwhere(out < 0)
        for node, dim in neg_idx:
            print(f"   ノード{node}・次元{dim}: {out[node, dim]:.3f}")

    return out[0]  # 道路1の予測 [t3, t4]


# === ログ出力クラス ===
class TrafficLogger:
    def __init__(self, used_road_ids):
        self.used_road_ids = used_road_ids
        self.measurement_count = 0
        self.prediction_count = 0
        
    def log_measurement_data(self, road_id, t1_value, t2_minus_t1_value, step):
        """計測データをログ出力"""
        self.measurement_count += 1
        if road_id in [5,6]:
            print(f"📊 [{self.measurement_count:3d}] 道路{road_id}: t1={t1_value:2d}, t2-t1={t2_minus_t1_value:2d} (step={step})")
    
    def log_prediction_trigger(self, all_data):
        """予測実行時の全道路データをログ出力"""
        print("🔍 === 予測実行時の全道路データ ===")
        for road_id in self.used_road_ids:
            if all_data[road_id] is not None:
                t1, t2_minus_t1 = all_data[road_id]
                print(f"   道路{road_id}: t1={t1:2d}, t2-t1={t2_minus_t1:2d}")
            else:
                print(f"   道路{road_id}: データなし")
        print("=====================================")
    
    def log_prediction_result(self, count, t1_count, diff, pred_t3, pred_t4, step):
        """予測結果をログ出力"""
        print(f"🔮 [{count:3d}回目] 入力: t1={t1_count:2d}, t2-t1={diff:2d} → 予測: t3={pred_t3:2d}, t4={pred_t4:2d} (step={step})")
    
    def log_data_incomplete(self, step):
        """データ不完全時のログ出力"""
        print(f"⏭️ 道路1予測パス（step={step}）: データ未揃い → 破棄")

# === パス設定 ===
sumocfg_path = r"C:\Users\tslab\Desktop\予測\町モデルデータ\toyama_shouwa.sumocfg"
log_dir_path = r"C:\Users\tslab\Desktop\予測\制御\csv掃き出し\各道路計測ログ"
train_csv_path = r"C:\Users\tslab\Desktop\予測\新環境待ち台数データ_12.22.csv"
adj_path = r"C:\Users\tslab\Desktop\予測\新環境_隣接行列.csv"
model_path = r"C:\Users\tslab\Desktop\予測\予測モデル\0. 制御用_待ち台数単体(3.22)\道路1\GCN_epoch_best_model.pth"

# === 使用ノード ===
used_road_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]
used_road_ids = [str(i+1) for i in range(9)]  # 道路1～9
latest_data = {rid: None for rid in used_road_ids}

# === ログ出力器初期化 ===
logger = TrafficLogger(used_road_ids)

# === スケーラー構築 ===
df_all = pd.read_csv(train_csv_path, encoding='cp932')
df_used = df_all.iloc[:, used_road_indices]
scaler = MinMaxScaler()
scaler.fit(df_used.values)

# === edge_index構築 ===
adj_matrix = pd.read_csv(adj_path, encoding='shift-jis', index_col=0).values
edge_index_raw = np.array(np.nonzero(adj_matrix)).astype(np.int64)
mask = np.isin(edge_index_raw[0], used_road_indices) & np.isin(edge_index_raw[1], used_road_indices)
edge_index_filtered = edge_index_raw[:, mask]
id_map = {old: new for new, old in enumerate(used_road_indices)}
edge_index_mapped = np.vectorize(id_map.get)(edge_index_filtered)
edge_index = torch.tensor(edge_index_mapped, dtype=torch.long)

# === 信号設定 ===
MAX_RECORD_COUNT = 720 # 記録する最大サイクル数（例：1日分）
# MAX_RECORD_COUNT = 300 # 記録する最大サイクル数（例：1日分）
PREP_TIME = 1200
# TRAFFIC_LIGHT_CONFIG = {
#     "A": [
#     {"id": "1", "edges": ["DtoA"], "red": "rrrryyyyyrrrryyyyy", "green": "rrrrrrrrrrrrrrrrrr"},
#     {"id": "10", "edges": ["-E70"], "red": "yyyyrrrrryyyyrrrrr", "green": "rrrrrrrrrrrrrrrrrr"},
#     {"id": "11", "edges": ["E2"], "red": "yyyyrrrrryyyyrrrrr", "green": "rrrrrrrrrrrrrrrrrr"}
#     ],
#     "D": [{"id": "2", "edges": ["E91", "E92"], "red": "rrryyyyyrrryyyyy", "green": "rrrrrrrrrrrrrrrr"}],
#     "E": [{"id": "3", "edges": ["E12"], "red": "rrryyyyyyyy", "green": "rrrrrrrrrrr"}],
#     "F": [{"id": "4", "edges": ["E159", "E158"], "red": "rrryyyyyrrryyyyy", "green": "rrrrrrrrrrrrrrrr"}],
#     "I": [{"id": "5", "edges": ["E26"], "red": "rrrryyyrrrryyy", "green": "rrrrrrrrrrrrrr"}],
#     "J": [
#         {"id": "9", "edges": ["M-1toJ"], "red": "rrrryyyyyrrrryyyyy", "green": "rrrrrrrrrrrrrrrrrr"},
#         {"id": "7", "edges": ["-E53", "-E35"], "red": "yyyyrrrrryyyyrrrrr", "green": "rrrrrrrrrrrrrrrrrr"},
#         {"id": "6", "edges": ["E196"], "red": "yyyyrrrrryyyyrrrrr", "green": "rrrrrrrrrrrrrrrrrr"}
#     ],
#     "K": [{"id": "8", "edges": ["E36"], "red": "yyyyyyrr", "green": "rrrrrrrr"}],
# }

# ゴリ押し用現示観測  (南, 東, 西, 北)
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
    {"id": "4", "edges": ["E57"], "red": "yyyyrrryyyyrrr", "green": "rrrrrrrrrrrrrr"},
    {"id": "42", "edges": ["155387430"], "red": "rrrryyyrrrryyy", "green": "rrrrrrrrrrrrrr"},          
    ],
    "J": [
    {"id": "1", "edges": ["E24", "-E23", "-E33"], "red": "rrrryyyyrrrryyyy", "green": "rrrrrrrrrrrrrrrr"}, #南
    {"id": "5", "edges": ["E37", "E36"], "red": "yyyyrrrryyyyrrrr","green": "rrrrrrrrrrrrrrrr"}, #西
    {"id": "6", "edges": ["-E25", "-E69"], "red": "yyyyrrrryyyyrrrr","green": "rrrrrrrrrrrrrrrr"}, #東
    {"id": "10", "edges": ["-E35", "E34", "-E1"], "red": "rrrryyyyrrrryyyy", "green": "rrrrrrrrrrrrrrrr"}
    ],
    "K": [
    {"id": "43", "edges": ["E71"], "red": "rrrrrryy", "green": "rrrrrrrr"},
    {"id": "7", "edges": ["155398822#14", "155398822#13"], "red": "yyyyyyrr", "green": "rrrrrrrr"},
    {"id": "44", "edges": ["E70"], "red": "yyyyyyrr", "green": "rrrrrrrr"}
    ],
    "L": [
    {"id": "45", "edges": ["E76"], "red": "rrrryyyrrrryyy", "green": "rrrrrrrrrrrrrr"},
    {"id": "46", "edges": ["-E74", "-E75"], "red": "yyyyrrryyyyrrr", "green": "rrrrrrrrrrrrrr"},
    {"id": "47", "edges": ["E73", "E72"], "red": "yyyyrrryyyyrrr", "green": "rrrrrrrrrrrrrr"},
    {"id": "48", "edges": ["E77"], "red": "rrrryyyrrrryyy", "green": "rrrrrrrrrrrrrr"}
    ],
    "M": [
    {"id": "8", "edges": ["E31", "E30"], "red": "rrryyyyrrryyyy", "green": "rrrrrrrrrrrrrr"},
    {"id": "49", "edges": ["155390062#6"], "red": "yyyrrrryyyrrrr", "green": "rrrrrrrrrrrrrr"},
    {"id": "50", "edges": ["155389284#1"], "red": "yyyrrrryyyrrrr", "green": "rrrrrrrrrrrrrr"},
    {"id": "51", "edges": ["543210000#15", "-E32"], "red": "rrryyyyrrryyyy", "green": "rrrrrrrrrrrrrr"}
    ],
    "N": [
    {"id": "9", "edges": ["E20", "E19"], "red": "rrryyyyrrryyyy", "green": "rrrrrrrrrrrrrr"},
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

# ログファイル設定
# === 予測結果ログ =====
prediction_log_path = r"C:\Users\tslab\Desktop\予測\制御\csv掃き出し\予測結果_道路1.csv"
prediction_log_file = open(prediction_log_path, "w", newline="", encoding="utf-8-sig")
prediction_writer = csv.writer(prediction_log_file)
prediction_writer.writerow(["回数", "step", "前半", "後半", "予測前半", "予測後半"])

# === フェーズ時間ログファイル ===
phase_log_path = r"C:\Users\tslab\Desktop\予測\制御\csv掃き出し\信号A_フェーズ時間ログ.csv"
phase_log_file = open(phase_log_path, "w", newline="", encoding="utf-8-sig")
phase_writer = csv.writer(phase_log_file)
phase_writer.writerow(["回数", "step", "パターンA(秒)", "パターンB(秒)", "予測t3", "予測t4"])

# === 無駄時間ログ ====
wasted_green_log_path = r"C:\Users\tslab\Desktop\予測\制御\csv掃き出し\無駄青時間ログ.csv"
wasted_green_file = open(wasted_green_log_path, "w", newline="", encoding="utf-8-sig")
wasted_green_writer = csv.writer(wasted_green_file)
wasted_green_writer.writerow(["回数", "step", "監視秒数", "無駄青時間(s)"])

# === 遅れ時間ログ ===
# A交差点
delay_log_path = r"C:\Users\tslab\Desktop\予測\制御\csv掃き出し\遅れ時間\信号機A_遅れ時間ログ_主道路(南).csv"
delay_log_file = open(delay_log_path, "w", newline="", encoding="utf-8-sig")
delay_writer = csv.writer(delay_log_file)
delay_writer.writerow(["回数", "step", "前半台数", "後半台数", "delta", "okure", "遅れ時間(秒)"])

noth_delay_log_path = r"C:\Users\tslab\Desktop\予測\制御\csv掃き出し\遅れ時間\信号機A_遅れ時間ログ_主道路(北).csv"
noth_delay_log_file = open(noth_delay_log_path, "w", newline="", encoding="utf-8-sig")
noth_delay_writer = csv.writer(noth_delay_log_file)
noth_delay_writer.writerow(["回数", "step", "前半台数", "後半台数", "delta", "okure", "遅れ時間(秒)"])

delay_minor_log_path = r"C:\Users\tslab\Desktop\予測\制御\csv掃き出し\遅れ時間\信号機A_遅れ時間_従道路.csv"
delay_minor_log_file = open(delay_minor_log_path, "w", newline="", encoding="utf-8-sig")
delay_minor_writer = csv.writer(delay_minor_log_file)
delay_minor_writer.writerow(["回数", "step", "道路ID", "前半台数", "後半台数", "delta", "okure", "遅れ時間(秒)"])

# 全道路
all_delay_log_path = r"C:\Users\tslab\Desktop\予測\制御\csv掃き出し\遅れ時間\全遅れ時間ログ_主道路.csv"
all_delay_log_file = open(all_delay_log_path, "w", newline="", encoding="utf-8-sig")
all_delay_writer = csv.writer(all_delay_log_file)
all_delay_writer.writerow(["回数", "step", "道路ID", "前半台数", "後半台数", "okure", "遅れ時間(秒)"])

all_delay_minor_log_path = r"C:\Users\tslab\Desktop\予測\制御\csv掃き出し\遅れ時間\全遅れ時間_従道路.csv"
all_delay_minor_log_file = open(all_delay_minor_log_path, "w", newline="", encoding="utf-8-sig")
all_delay_minor_writer = csv.writer(all_delay_minor_log_file)
all_delay_minor_writer.writerow(["回数", "step", "道路ID", "前半台数", "後半台数", "okure", "遅れ時間(秒)"])

green_log_path = r"C:\Users\tslab\Desktop\予測\制御\csv掃き出し\green_durations.csv"
green_log_file = open(green_log_path, "w", newline="", encoding="utf-8-sig")
green_writer = csv.writer(green_log_file)
green_writer.writerow(["回数", "step", "種類", "青時間(秒)"])  # 種類 = 主道路 or 従道路


# === 道路[1, 12], 道路[10, 11] 待ち台数 & 遅れ時間ログ ===
queue_delay_log_path = r"C:\Users\tslab\Desktop\予測\制御\csv掃き出し\queue_and_delay.csv"
queue_delay_file = open(queue_delay_log_path, "w", newline="", encoding="utf-8-sig")
queue_delay_writer = csv.writer(queue_delay_file)
queue_delay_writer.writerow(["回数", "step", "道路ID", "待ち台数", "合計遅れ時間(秒)"])

# ==== 重道路計測用ファイル指定 ================
def load_green_rows(csv_path):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    rows = df[df["種類"] == "主道路"][["step", "青時間(秒)"]].to_dict("records")
    return rows



# 初期時間を固定
INITIAL_PHASE_DURATIONS = {
    0: 60,  # フェーズ0：rrrrgGGGgrrrrgGGGg
    6: 26   # フェーズ3：gGGgrrrrrgGGgrrrrr
}

def adjust_signal_A_based_on_prediction(t3, t4, step, prediction_count):
    try:
        light_id = "J"
        logic = traci.trafficlight.getAllProgramLogics(light_id)[0]
        old_phases = logic.getPhases()

        if len(old_phases) != 6:
            print(f"⚠️ 信号機Aのフェーズ数が想定と異なります: {len(old_phases)} フェーズ")
            return

        # === 台数に応じた信号時間調整 ===
        # delta = (t3 + t4) * 4   # ← 変更ポイント（大小関係を無視して合計台数×4秒）

        # # 安全範囲を制限（必要なら調整）
        # delta = max(-10, min(delta, 60))
        
        if t3 > t4:
            # 1台ごとに2秒延長（6台超過分など関係なく）
            delta = t3 * 1  # 台数差 × 2 秒（上限なし）
            # minor_delta = -2

        elif t3 < t4:
            delta = t4 * 1
            # minor_delta = t4 * -2
        else:
            delta = 0

        delta = max(-10, min(delta, 20))  # 安全範囲に制限

        # # 安全範囲を制限（最大+20秒まで）
        # delta = min(delta, 20)

        global last_phase_delta, signal_A_adjusted_once
        last_phase_delta = delta
        signal_A_adjusted_once = True  # 初めて制御されたらTrueにする

        from traci._trafficlight import Phase

        new_phases = []
        for i, phase in enumerate(old_phases):
            if i == 0:
                dur = INITIAL_PHASE_DURATIONS[0] + delta
            elif i == 6: 
                # dur = max(5, INITIAL_PHASE_DURATIONS[3] - delta)
                dur =  INITIAL_PHASE_DURATIONS[3]
            else:
                dur = phase.duration

            min_dur = getattr(phase, "minDur", dur)
            max_dur = getattr(phase, "maxDur", dur)
            new_phases.append(Phase(duration=dur, state=phase.state, minDur=min_dur, maxDur=max_dur))

        new_logic = traci.trafficlight.Logic(
            programID=logic.programID,
            type=logic.type,
            currentPhaseIndex=logic.currentPhaseIndex,
            phases=new_phases
        )

        traci.trafficlight.setProgramLogic(light_id, new_logic)

        print(f"✅ [制御] 信号A更新 (回数={prediction_count}): フェーズ0→{new_phases[0].duration}s, フェーズ3→{new_phases[3].duration}s")

        # === ログにフェーズ時間を出力（記録）===
        if "phase_writer" in globals():
            phase_writer.writerow([
                prediction_count,
                step,
                INITIAL_PHASE_DURATIONS[0] + delta,
                max(5, INITIAL_PHASE_DURATIONS[3]),
                t3,
                t4
            ])

    except Exception as e:
        print(f"⚠️ 信号A制御中エラー: {e}")


# ===============================
# 　　　　ギャップ感応制御
# ===============================
class GapCycleController:
    def __init__(self, tls_id):
        self.tls_id = tls_id

        self.MAIN_GREEN = 0
        self.MINOR_GREEN = 6

        # 主道路
        self.MIN_MAIN_GREEN = 50
        self.MAX_MAIN_GREEN = 70
        self.BASE_MAIN_GREEN = 60
        self.MAX_GAP = 5

        self.MIN_MAIN_GREEN_2 = 50


        # 従道路（固定）
        self.FIXED_MINOR_GREEN = 26

        self.MAJOR_DETECTORS = ["J_south_0", "J_south_1", "J_south_2"]

        # ===== 内部状態 =====
        self.prev_phase = None

        # 主道路「青」の状態管理
        self.main_green_active = False
        self.main_green_start_time = None

        # ギャップ管理
        self.last_vehicle_time = 0

        # サイクル管理
        self.cycle_index = 0      # 0: 1サイクル目（ギャップ）, 1: 2サイクル目（固定）
        self.main_green_1 = None
        self.target_main_green = None

    # ===============================================
    # ギャップ感応制御 - フェーズ duration を変更する
    # ===============================================
    def set_main_green_duration(self, duration):
        logic = traci.trafficlight.getAllProgramLogics(self.tls_id)[0]
        phases = logic.getPhases()

        phases[self.MAIN_GREEN].duration = duration
        phases[self.MAIN_GREEN].minDur = duration
        phases[self.MAIN_GREEN].maxDur = duration

        new_logic = traci.trafficlight.Logic(
            programID=logic.programID,
            type=logic.type,
            currentPhaseIndex=logic.currentPhaseIndex,
            phases=phases
        )
        traci.trafficlight.setProgramLogic(self.tls_id, new_logic)

    # ===============================
    # ギャップ感応制御 - ギャップ判定
    # ===============================
    def gap_occurred(self, step):
        for det in self.MAJOR_DETECTORS:
            if traci.inductionloop.getLastStepVehicleNumber(det) > 0:
                self.last_vehicle_time = step
                return False
        return (step - self.last_vehicle_time) >= self.MAX_GAP

    # ===============================
    # ギャップ感応制御 - 毎ステップ呼び出す
    # ===============================
    def gap_occurred(self, step):
        for det in self.MAJOR_DETECTORS:
            if traci.inductionloop.getLastStepVehicleNumber(det) > 0:
                self.last_vehicle_time = step
                return False
        return (step - self.last_vehicle_time) >= self.MAX_GAP

    # -------------------------------------------------
    # 毎ステップ呼び出し
    # -------------------------------------------------
    def update(self, step):
        phase = traci.trafficlight.getPhase(self.tls_id)

        # =====================================
        # 主道路「青」開始の検出
        # =====================================
        if phase == self.MAIN_GREEN and not self.main_green_active:
            self.main_green_active = True
            self.main_green_start_time = step
            self.last_vehicle_time = step

        # =====================================
        # 主道路「青」終了の検出（純粋な青時間）
        # =====================================
        elif phase != self.MAIN_GREEN and self.main_green_active:
            green_time = step - self.main_green_start_time
            self.main_green_active = False

            # ---------- 1サイクル目終了 ----------
            if self.cycle_index == 0:
                self.main_green_1 = green_time

                # ★ 指定どおりの式 ★
                self.target_main_green = (
                    2 * self.BASE_MAIN_GREEN - self.main_green_1
                )

                # 安全制限
                self.target_main_green = max(
                    self.MIN_MAIN_GREEN_2,
                    min(self.MAX_MAIN_GREEN, self.target_main_green)
                )

                # ★ 次の主道路フェーズの duration を書き換える ★
                self.set_main_green_duration(self.target_main_green)

                self.cycle_index = 1

            # ---------- 2サイクル目終了 ----------
            else:
                # 次は再びギャップ感応に戻す
                self.cycle_index = 0

        # =====================================
        # 主道路が青の間の制御
        # =====================================
        if phase == self.MAIN_GREEN and self.main_green_active:
            green_time = step - self.main_green_start_time

            # ---------- 2サイクル目 ----------
            # → SUMO が duration 通りに進むので何もしない
            if self.cycle_index == 1:
                pass

            # ---------- 1サイクル目（ギャップ感応） ----------
            else:
                gap = self.gap_occurred(step)

                # 最大青
                if green_time >= self.MAX_MAIN_GREEN:
                    traci.trafficlight.setPhase(self.tls_id, phase + 1)

                # ギャップ終了
                elif green_time >= self.MIN_MAIN_GREEN and gap:
                    traci.trafficlight.setPhase(self.tls_id, phase + 1)



def get_measurement_times_for_main_road(sim_time, base_t1, base_t2, delta_adjustment=0):
    """
    主道路（道路1）の赤信号 前半/後半測定時刻を返す。

    主道路の青信号を延長すると、その分だけ赤信号が短くなる。
    t1（前半）は前倒し、t2（後半）も前倒しで調整。

    Parameters:
        sim_time (float): 現在のシミュレーション時刻
        base_t1 (int): 通常の赤信号前半時間（秒）
        base_t2 (int): 通常の赤信号後半時間（秒）
        delta_adjustment (int): 主道路の青信号延長時間（秒）

    Returns:
        (int, int): t1, t2 の測定タイミング
    """
    
    delta_adjustment = 0

    t1_time = int(sim_time + base_t1 - delta_adjustment // 2)
    t2_time = int(sim_time + base_t1 + base_t2 - delta_adjustment)
    return t1_time, t2_time

def get_next_green_and_times(sim_time, green_rows, index, tolerance=1):
    """
    デバッグ版：
    ・新規計測 / キャッシュ再利用 を必ずログ出力
    """

    # ---- 初期化 ----
    if not hasattr(get_next_green_and_times, "_cache"):
        get_next_green_and_times._cache = {
            "used": False,
            "t1": None,
            "t2": None,
            "last_sim_time": None
        }

    cache = get_next_green_and_times._cache

    # ---- sim_time が変わったら新イベント扱い ----
    if cache["last_sim_time"] != sim_time:
        cache["used"] = False
        cache["last_sim_time"] = sim_time

    # ---- キャッシュ再利用 ----
    if cache["used"]:
        print(
            f"🔁 [再利用] sim_time={sim_time} "
            f"t1={cache['t1']}, t2={cache['t2']} (index={index})"
        )
        return cache["t1"], cache["t2"], index

    # ---- 新規CSV消費 ----
    if index >= len(green_rows):
        raise IndexError("❌ 青時間CSVをすべて消費しました")

    row = green_rows[index]
    step = int(row["step"])
    green = int(row["青時間(秒)"])

    print(
        f"🆕 [新規計測] sim_time={sim_time}, "
        f"csv_index={index}, csv_step={step}, green={green}"
    )

    if abs(sim_time - step) > tolerance:
        print(
            f"⚠️ stepずれ: csv_step={step}, sim_time={sim_time}, diff={sim_time-step}"
        )

    t1_time = int(sim_time + (green + 28) / 2)
    t2_time = int(sim_time + green + 26)

    cache["t1"] = t1_time
    cache["t2"] = t2_time
    cache["used"] = True

    print(
        f"📏 [計測確定] t1={t1_time}, t2={t2_time}"
    )

    return t1_time, t2_time, index + 1





# ==== 時間距離図 =============================================================
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd

# パラメータ
MEASURE_START = 30050  # 計測開始秒
MEASURE_END   = 30800  # 計測終了秒
MAX_MEASURE_VEHICLES = None  # Noneなら無制限、数値なら台数制限

DEFAULT_INTERVAL = 120
DEFAULT_MAX_N = 5

# 区間長（交差点内は無視、0mで扱わない）
EDGE_LENGTHS = {
    "M-1toJ": 74,
    "E158": 100,
    "E159": 76,
    "E12": 220,
    "-543210000#5": 31,
    "E92": 87,
    "E91": 36,
    "DtoA": 123,
    "E1": 200,

    "-E1": 200,
    "-E90": 123,
    "-E91": 36,
    "-E92": 87,
    "543210000#5": 31,
    "-E12": 220,
    "-E159": 76,
    "-E158": 100,
    "543210000#13": 74,
}

SIGNAL_INTERSECTIONS = {
    "A交差点": (EDGE_LENGTHS["M-1toJ"] + EDGE_LENGTHS["E158"] + EDGE_LENGTHS["E159"] + EDGE_LENGTHS["E12"] + EDGE_LENGTHS["-543210000#5"] + EDGE_LENGTHS["E92"] + EDGE_LENGTHS["E91"]+ EDGE_LENGTHS["DtoA"]),
    "D交差点": (EDGE_LENGTHS["M-1toJ"] + EDGE_LENGTHS["E158"] + EDGE_LENGTHS["E159"] + EDGE_LENGTHS["E12"] + EDGE_LENGTHS["-543210000#5"] + EDGE_LENGTHS["E92"] + EDGE_LENGTHS["E91"]),
    "E交差点": (EDGE_LENGTHS["M-1toJ"] + EDGE_LENGTHS["E158"] + EDGE_LENGTHS["E159"] + EDGE_LENGTHS["E12"]),
    "F交差点": (EDGE_LENGTHS["M-1toJ"] + EDGE_LENGTHS["E158"] + EDGE_LENGTHS["E159"]),
    "J交差点": (EDGE_LENGTHS["M-1toJ"]),
}

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
    for tl_id, name in [("A","A交差点"), ("D","D交差点"), ("E","E交差点"),
                        ("F","F交差点"), ("J","J交差点")]:
        current_phase = traci.trafficlight.getPhase(tl_id)
        red_now = (2 <= current_phase <= 5)

        if red_now:
            if not red_active_states.get(name, False):
                # 赤開始
                signal_red_intervals[name].append([sim_time, None])
                red_active_states[name] = True
        else:
            if red_active_states.get(name, False):
                # 赤終了
                if signal_red_intervals[name]:
                    signal_red_intervals[name][-1][1] = sim_time
                red_active_states[name] = False

    # === 北方向 spawn 検知 ===
    for veh_id in traci.simulation.getDepartedIDList():
        edge_id = traci.vehicle.getRoadID(veh_id)
        if edge_id == "-E1" and veh_id not in enter_time_north:
            if (MAX_MEASURE_VEHICLES is None) or (len(measured_north) < MAX_MEASURE_VEHICLES):
                enter_time_north[veh_id] = sim_time
                measured_north.add(veh_id)

    # === 南方向 M-1toJ進入を検知 ===
    for veh_id in traci.vehicle.getIDList():
        edge_id = traci.vehicle.getRoadID(veh_id)

        if veh_id not in enter_time_south and edge_id == "M-1toJ":
            if (MAX_MEASURE_VEHICLES is None) or (len(measured_south) < MAX_MEASURE_VEHICLES):
                enter_time_south[veh_id] = sim_time
                measured_south.add(veh_id)

        # 南方向の記録
        if veh_id in measured_south and edge_id in [
            "M-1toJ", "E158", "E159", "E12",
            "-543210000#5", "E92", "E91", "DtoA", "E1"
        ]:
            pos = traci.vehicle.getLanePosition(veh_id)
            vehicle_traces[veh_id].append(("south", sim_time, edge_id, pos))
            if edge_id == "E1":
                completed_south.add(veh_id)

        # 北方向の記録
        if veh_id in measured_north and edge_id in [
            "-E1", "-E90", "-E91", "-E92",
            "543210000#5", "-E12", "-E159","-E158","543210000#13"
        ]:
            pos = traci.vehicle.getLanePosition(veh_id)
            vehicle_traces[veh_id].append(("north", sim_time, edge_id, pos))
            if edge_id == "543210000#13":
                completed_north.add(veh_id)


# ===============================
# 座標変換
# ===============================
def convert_x_south(row):
    order = ["M-1toJ", "E158", "E159", "E12", "-543210000#5", "E92", "E91", "DtoA", "E1"]
    x = 0
    for edge in order:
        if row["edge"] == edge:
            return x + row["pos"]
        x += EDGE_LENGTHS[edge]
    return None

def convert_x_north(row):
    order = ["-E1", "-E90", "-E91", "-E92", "543210000#5", "-E12", "-E159","-E158","543210000#13"]
    x = sum(EDGE_LENGTHS[e] for e in order)
    for edge in order:
        x -= EDGE_LENGTHS[edge]
        if row["edge"] == edge:
            return x + (EDGE_LENGTHS[edge] - row["pos"])
    return None


# ===============================
# プロット関数
# ===============================
def plot_time_space_diagram():
    plt.figure(figsize=(14, 7))

    def draw_traces(df, color):
        plt.plot(df["x"], df["sim_time"], color=color, alpha=0.4)

    # 南方向
    for veh_id in completed_south:
        if veh_id not in enter_time_south:
            continue
        records = vehicle_traces[veh_id]
        if not records:
            continue
        df = pd.DataFrame(records, columns=["dir", "sim_time", "edge", "pos"])
        df["x"] = df.apply(convert_x_south, axis=1)
        draw_traces(df, "blue")

    # 北方向
    for veh_id in completed_north:
        if veh_id not in enter_time_north:
            continue
        records = vehicle_traces[veh_id]
        if not records:
            continue
        df = pd.DataFrame(records, columns=["dir", "sim_time", "edge", "pos"])
        df["x"] = df.apply(convert_x_north, axis=1)
        draw_traces(df, "green")

    # 軸ラベル
    plt.xlabel("リンク距離 [m]")
    plt.ylabel("シミュレーション時間 [s]")
    plt.title("時間距離図（信号制御付き, シミュレーション時間基準）")

    # 主目盛り
    main_positions = [
        0,
        EDGE_LENGTHS["M-1toJ"],
        EDGE_LENGTHS["M-1toJ"] + EDGE_LENGTHS["E158"] + EDGE_LENGTHS["E159"],
        EDGE_LENGTHS["M-1toJ"] + EDGE_LENGTHS["E158"] + EDGE_LENGTHS["E159"] + EDGE_LENGTHS["E12"],
        EDGE_LENGTHS["M-1toJ"] + EDGE_LENGTHS["E158"] + EDGE_LENGTHS["E159"] + EDGE_LENGTHS["E12"] + EDGE_LENGTHS["-543210000#5"] + EDGE_LENGTHS["E92"] + EDGE_LENGTHS["E91"],
        SIGNAL_INTERSECTIONS["A交差点"],
        SIGNAL_INTERSECTIONS["A交差点"] + EDGE_LENGTHS["E1"]
    ]
    main_labels = ["南端", "J交差点", "F交差点", "E交差点", "D交差点", "A交差点", "北端"]

    plt.xticks(main_positions, main_labels)
    plt.grid(True, which="major", linestyle="--", alpha=0.6)

    # -------------------------------
    # 赤信号時間帯の描画（sim_time基準）
    # -------------------------------
    for name, x_pos in SIGNAL_INTERSECTIONS.items():
        if name in signal_red_intervals:
            for start, end in signal_red_intervals[name]:
                if end is None:  # まだ赤のまま終わってない場合
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


last_phase_delta = None  # グローバルに定義（run_simulationの前）
signal_A_adjusted_once = False  # 初期化：制御が1度も入っていない

green_monitoring_active = False
green_monitor_start_time = None
green_monitor_end_time = None
green_monitor_no_pass_seconds = 0
green_monitor_phase_end_time = None

# === 青時間記録用グローバル ===
main_green_durations = []
minor_green_durations = []


# ===============================
# シミュレーション開始
# ===============================
def run_simulation(sumocfg_path, log_dir_path, control_signal=True):
    global prediction_count, last_phase_delta, signal_A_adjusted_once
    global green_monitoring_active, green_monitor_start_time
    global green_monitor_end_time, green_monitor_no_pass_seconds
    global green_monitor_phase_end_time

    if not os.path.exists(log_dir_path):
        os.makedirs(log_dir_path)

    # === スケーラーの読み込み ===
    scaler_path = r"C:\Users\tslab\Desktop\予測\予測モデル\0. 制御用_待ち台数単体(3.22)\道路1\scaler.pkl"
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"❌ スケーラーファイルが存在しません: {scaler_path}")

    scaler = joblib.load(scaler_path)
    print("✅ スケーラー読み込み完了")

    sumoBinary = sumolib.checkBinary('sumo-gui')
    traci.start([sumoBinary, "-c", sumocfg_path, "--start", "--quit-on-end"])
    sim_time = 0
    prev_states = defaultdict(dict)
    pending_counts = defaultdict(list)
    first_half_storage = defaultdict(dict)

    writers = {}
    scheduled_times_A = set()
    last_phase_delta_before_prediction = 0  # 予測直前のdeltaを記録

    vehicle_delay_time = {}  # {veh_id: 累積停止時間[s]}

    queue_delay_writers = {}
    vehicle_delay_time = {}
    cycle_buffer = {rid: {"queue_sum": 0, "delay_sum": 0} for rid in ["1", "12", "10", "11"]}
    # サイクル中に停車した車両IDを記録するセット
    cycle_stopped_vehicles = {rid: set() for rid in ["1", "5", "6", "10"]}

    # === 重道路計測用 ==========================
    green_rows = load_green_rows(r"C:\Users\tslab\Desktop\予測\制御\現示時間\主道路.csv")
    green_row_index = 0


    # === 交差点A出力フォルダ ===
    a_dir = os.path.join(log_dir_path, "交差点A")
    os.makedirs(a_dir, exist_ok=True)

    # === 交差点A全体の合計ログ ===
    total_file_path = os.path.join(a_dir, "交差点A_total_queue_delay.csv")
    total_file = open(total_file_path, "w", newline="", encoding="utf-8-sig")
    total_writer = csv.writer(total_file)
    total_writer.writerow(["サイクル", "step", "全道路停車車両数合計", "全道路遅れ時間合計"])


    # 個別道路用（集計用バッファには残す）
    cycle_stopped_vehicles = {rid: set() for rid in ["1", "5", "6", "10"]}
    vehicle_delay_time = {}

    queue_delay_files = {}     # ← ファイルオブジェクトを保持
    queue_delay_writers = {}   # ← writer を保持

    for road_id in ["1", "5", "6", "10"]:
        file_path = os.path.join(a_dir, f"道路{road_id}_queue_delay.csv")
        f = open(file_path, "w", newline="", encoding="utf-8-sig")
        writer = csv.writer(f)
        writer.writerow(["サイクル", "step", "道路ID", "待ち台数合計", "遅れ時間合計"])
        queue_delay_files[road_id] = f
        queue_delay_writers[road_id] = writer


    # サイクルごとに一時的に待ち台数・遅れ時間を溜めるバッファ
    cycle_queue_counts = {rid: [] for rid in ["1", "12", "10", "11"]}
    cycle_delay_sums   = {rid: [] for rid in ["1", "12", "10", "11"]}
    cycle_counter = 0  # サイクル番号

    cycle_counter = 0
    last_phase_A = None

    # === 青時間ログ用 ===
    main_green_durations = []
    minor_green_durations = []
    green_start_time = None
    current_green_type = None  # "main" or "minor"

    cycle_count = 0
    green_state = {"start": None, "type": None}

    tls_id = "A"

    # 例: フェーズ番号で分類（あなたのtlLogicに合わせて調整してください）
    MAIN_GREEN_PHASES = [0]   # 主道路が青になるフェーズ番号
    MINOR_GREEN_PHASES = [6]  # 従道路が青になるフェーズ番号

    for tl_id, configs in TRAFFIC_LIGHT_CONFIG.items():
        for conf in configs:
            road_id = conf["id"]
            f = open(os.path.join(log_dir_path, f"道路{road_id}.csv"), "w", newline="", encoding="utf-8")
            writer = csv.writer(f)
            writer.writerow(["step", "count"])
            writers[road_id] = writer

    print("🚦 シミュレーション開始...")

    gap_controller_A = GapCycleController("J")

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        sim_time = traci.simulation.getTime()

        # ギャップ感応制御用呼び出し
        gap_controller_A.update(sim_time)

        phase = traci.trafficlight.getPhase("J")

        if phase in MAIN_GREEN_PHASES or phase in MINOR_GREEN_PHASES:
            if green_state["start"] is None:
                green_state["start"] = sim_time
                green_state["type"] = "主道路" if phase in MAIN_GREEN_PHASES else "従道路"
        else:
            if green_state["start"] is not None:
                duration = sim_time - green_state["start"]
                cycle_count += 1
                green_writer.writerow([cycle_count, sim_time, green_state["type"], duration])
                print(f"🟢 {green_state['type']} 青 {duration} 秒 (step={sim_time})")
                green_state["start"] = None
                green_state["type"] = None

        # === サイクル判定（交差点Jの信号で） ===
        current_phase_A = traci.trafficlight.getPhase("J")
        total_phases_A = len(traci.trafficlight.getAllProgramLogics("J")[0].getPhases())

        if last_phase_A == total_phases_A - 1 and current_phase_A == 0 and sim_time >= 1200:
            cycle_counter += 1

            total_stopped = 0
            total_delay = 0

            for road_id in ["1", "5", "6", "10"]:
                stopped_count = len(cycle_stopped_vehicles[road_id])  # 🚗 ユニーク車両数
                delay_sum = sum(vehicle_delay_time[veh] for veh in cycle_stopped_vehicles[road_id])  # 遅れ時間合計

                queue_delay_writers[road_id].writerow(
                    [cycle_counter, sim_time, road_id, stopped_count, delay_sum]
                )
                # print(f"📤 サイクル{cycle_counter} 道路{road_id}: 停車車両数={stopped_count}, 遅れ時間={delay_sum}")

                # === 全体合計に加算 ===
                total_stopped += stopped_count
                total_delay += delay_sum

            # === 交差点A全体の出力 ===
            total_writer.writerow([cycle_counter, sim_time, total_stopped, total_delay])
            # print(f"📊 サイクル{cycle_counter} 交差点A合計: 停車車両数={total_stopped}, 遅れ時間={total_delay}")

            # 次サイクル用にリセット
            cycle_stopped_vehicles = {rid: set() for rid in ["1", "5", "6", "10"]}

        last_phase_A = current_phase_A


        # === 時間距離図 ===
        log_vehicle_positions(sim_time)


        # === 対象道路の待ち台数と遅れ時間を記録 ===
        if sim_time >= 1200:
            for road_id in ["1", "5", "6", "10"]:
                edges = None
                for tl_id, configs in TRAFFIC_LIGHT_CONFIG.items():
                    for conf in configs:
                        if conf["id"] == road_id:
                            edges = conf["edges"]
                            break
                    if edges: break
                if not edges:
                    continue

                for edge in edges:
                    for veh_id in traci.edge.getLastStepVehicleIDs(edge):
                        if traci.vehicle.getSpeed(veh_id) <= 0:
                            # サイクル内で停車した車両を記録
                            cycle_stopped_vehicles[road_id].add(veh_id)

                            # 遅れ時間は累積秒数を加算
                            if veh_id not in vehicle_delay_time:
                                vehicle_delay_time[veh_id] = 0
                            vehicle_delay_time[veh_id] += 1



        # === 無駄青監視：全青信号フェーズ末尾の固定秒数を調べる ===
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
                phase_log_file.close()
                wasted_green_file.close()
                return
            
            wasted_green_writer.writerow([
                prediction_count,
                green_monitor_phase_end_time,
                green_monitor_end_time - green_monitor_start_time,  # 実際の監視時間
                green_monitor_no_pass_seconds
            ])
            # print(f"🟩 [青信号全体監視] step={green_monitor_phase_end_time}, "
            #     f"青信号秒数={monitoring_duration}s, 無駄青={green_monitor_no_pass_seconds}s")
            green_monitoring_active = False
        # =====================================================

        for tl_id, configs in TRAFFIC_LIGHT_CONFIG.items():
            for i, config in enumerate(configs):
                try:
                    key = (tl_id, i)
                    current = traci.trafficlight.getRedYellowGreenState(tl_id)
                    prev = prev_states[tl_id].get(i, "")

                    if sim_time >= PREP_TIME and prev == config["red"] and current == config["green"]:
                        road_id = config["id"]
                        edges = config["edges"]

                        # === 無駄青（通常制御でも常時監視） ===
                        if config["edges"] == ["DtoA"]:
                            logic = traci.trafficlight.getAllProgramLogics("J")[0]
                            phase_0_duration = logic.getPhases()[0].duration  # ← 定義通りの青信号秒数（例：79）

                            green_monitor_start_time = int(sim_time)
                            green_monitor_end_time = int(sim_time + phase_0_duration)
                            green_monitor_phase_end_time = green_monitor_end_time
                            green_monitor_no_pass_seconds = 0
                            green_monitoring_active = True

                        # 測定時間設定 (A交差点のみ制御設定あり)
                        if road_id in ["1", "10"]:
                            if road_id in ["1", "10"]:
                                delta = last_phase_delta if signal_A_adjusted_once else 0
                                t1, t2 = get_measurement_times_for_main_road(sim_time, base_t1=28, base_t2=27, delta_adjustment=delta)
                                
                                # minor_delta = last_phase_delta if signal_A_adjusted_once else 0
                                # t1, t2 = get_measurement_times_for_main_road(sim_time, base_t1=21, base_t2=20, delta_adjustment=minor_delta)
                                expected_diff = t2 - t1
                                pending_counts[key] = [
                                    (t1, "t1", road_id, edges, expected_diff),
                                    (t2, "t2", road_id, edges, expected_diff)
                                ]
                        
                        elif road_id in ["5", "6"]:
                            t1, t2, green_row_index = get_next_green_and_times(
                                sim_time,
                                green_rows,
                                green_row_index
                            )

                            expected_diff = t2 - t1

                            # ---- 即時計測台数（デバッグ用）----
                            current_count = 0
                            for edge in edges:
                                for veh_id in traci.edge.getLastStepVehicleIDs(edge):
                                    if traci.vehicle.getSpeed(veh_id) <= 0:
                                        current_count += 1

                            # ---- デバッグログ ----
                            print(
                                f"📐 [計測予約] sim_time={sim_time}, road={road_id}, "
                                f"t1={t1}, t2={t2}, diff={expected_diff}, "
                                f"current_stop={current_count}, green_row_index={green_row_index}"
                            )

                            # ---- 通常処理 ----
                            pending_counts[key].append((t1, "t1", road_id, edges, expected_diff))
                            pending_counts[key].append((t2, "t2", road_id, edges, expected_diff))



                        elif road_id in ["11", "14"]:
                            t1 = int(sim_time + 27)
                            t2 = int(sim_time + 54)
                            pending_counts[key].append((t1, "t1", road_id, edges, 27))
                            pending_counts[key].append((t2, "t2", road_id, edges, 27))

                        elif road_id in ["12", "13"]:
                            t1 = int(sim_time + 44)
                            t2 = int(sim_time + 88)
                            pending_counts[key].append((t1, "t1", road_id, edges, 44))
                            pending_counts[key].append((t2, "t2", road_id, edges, 44))

                        elif road_id in ["15", "18", "19", "22", "24", "25", "27", "29", "30", "32", "35", "36", "39", "40", "42", "43", "45", "48", "49", "50", "52", "53", "56", "57", "60", "61"]:
                            t1 = int(sim_time + 42)
                            t2 = int(sim_time + 83)
                            pending_counts[key].append((t1, "t1", road_id, edges, 41))
                            pending_counts[key].append((t2, "t2", road_id, edges, 41))

                        else:
                            t1 = int(sim_time + 21)
                            t2 = int(sim_time + 41)
                            pending_counts[key].append((t1, "t1", road_id, edges, 20))
                            pending_counts[key].append((t2, "t2", road_id, edges, 20))

                    new_pending = []
                    for record in pending_counts[key]:
                        record_time, tag, road_id, edges, expected_diff = record
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

                                        # 👇 ここでマイナスチェックを追加
                                        if diff < 0:
                                            print(f"⚠️ 計測異常: 道路{road_id} の差分が負です。t1={prev_count}, t2={count} → 差={diff} (step={record_time})")

                                        if road_id in ["5", "6"]:
                                            delta_for_log = last_phase_delta_before_prediction if control_signal else 0
                                            okure = 91 + delta_for_log
                                            delay_seconds = prev_count * (okure * 0.75) + diff * (okure * 0.25)

                                            delay_minor_writer.writerow([
                                                prediction_count,
                                                record_time,
                                                road_id,
                                                prev_count,
                                                diff,
                                                delta_for_log,
                                                okure,
                                                int(round(delay_seconds))
                                            ])
                                        
                                        if road_id == "10":
                                            delta_for_log = last_phase_delta_before_prediction if control_signal else 0
                                            okure = 57 - delta_for_log
                                            delay_seconds = prev_count * (okure * 0.75) + diff * (okure * 0.25)

                                            noth_delay_writer.writerow([
                                                prediction_count,
                                                record_time,
                                                prev_count,
                                                diff,
                                                delta_for_log,
                                                okure,
                                                int(round(delay_seconds))
                                            ])
                                        
                                        if road_id in ["11", "14"]:
                                            okure = 57
                                            delay_seconds = prev_count * (okure * 0.75) + diff * (okure * 0.25)

                                            all_delay_writer.writerow([
                                                prediction_count,
                                                record_time,
                                                road_id,
                                                prev_count,
                                                diff,
                                                okure,
                                                int(round(delay_seconds))
                                            ])
                                        
                                        if road_id in ["12", "13"]:
                                            okure = 91
                                            delay_seconds = prev_count * (okure * 0.75) + diff * (okure * 0.25)

                                            all_delay_writer.writerow([
                                                prediction_count,
                                                record_time,
                                                road_id,
                                                prev_count,
                                                diff,
                                                okure,
                                                int(round(delay_seconds))
                                            ])

                                        if road_id in ["16", "17", "20", "21", "23", "26", "2", "28", "3", "31", "33", "34", "37", "38", "41", "4", "44", "7", "46", "47", "8", "51", "9", "54", "55", "58", "59", "62"]:
                                            okure = 42
                                            delay_seconds = prev_count * (okure * 0.75) + diff * (okure * 0.25)

                                            all_delay_writer.writerow([
                                                prediction_count,
                                                record_time,
                                                road_id,
                                                prev_count,
                                                diff,
                                                okure,
                                                int(round(delay_seconds))
                                            ])
                                        
                                        if road_id in ["15", "18", "19", "22", "24", "25", "27", "29", "30", "32", "35", "36", "39", "40", "42", "43", "45", "48", "49", "50", "52", "53", "56", "57", "60", "61"]:
                                            okure = 84
                                            delay_seconds = prev_count * (okure * 0.75) + diff * (okure * 0.25)

                                            all_delay_minor_writer.writerow([
                                                prediction_count,
                                                record_time,
                                                road_id,
                                                prev_count,
                                                diff,
                                                okure,
                                                int(round(delay_seconds))
                                            ])

                                        writers[road_id].writerow([record_time, diff])
                                        del first_half_storage[road_id][t1_time]
                                        latest_data[road_id] = (prev_count, diff)
                                        logger.log_measurement_data(road_id, prev_count, diff, record_time)

                                        if road_id == "1":
                                            last_phase_delta_before_prediction = last_phase_delta if last_phase_delta is not None else 0
                                            if all(latest_data[rid] is not None for rid in used_road_ids):
                                                logger.log_prediction_trigger(latest_data)
                                                t1s = [latest_data[rid][0] for rid in used_road_ids]
                                                dts = [latest_data[rid][1] for rid in used_road_ids]

                                                pred = run_prediction(
                                                    t1_array=t1s,
                                                    delay_array=dts,
                                                    edge_index=edge_index,
                                                    scaler=scaler,
                                                    model_path=model_path,
                                                    prediction_count=prediction_count
                                                )

                                                pred_t3, pred_t4 = map(int, np.round(pred))
                                                t1_count = latest_data["1"][0]
                                                diff = latest_data["1"][1]

                                                prediction_count += 1
                                                logger.log_prediction_result(prediction_count, t1_count, diff, pred_t3, pred_t4, record_time)
                                                if prediction_count >= MAX_RECORD_COUNT:
                                                    print("🛑 記録上限に到達。シミュレーション終了。")
                                                    traci.close()
                                                    phase_log_file.close()
                                                    wasted_green_file.close()
                                                    return

                                                prediction_writer.writerow([prediction_count, record_time, t1_count, diff, pred_t3, pred_t4])

                                                # delta: 信号制御が無効なら 0 にする
                                                if control_signal:
                                                    delta_for_log = last_phase_delta if last_phase_delta is not None else 0
                                                else:
                                                    delta_for_log = 0

                                                okure = 57 - delta_for_log
                                                delay_seconds = t1_count * (okure * 0.75) + diff * (okure * 0.25)

                                                delay_writer.writerow([
                                                    prediction_count,
                                                    record_time,
                                                    t1_count,
                                                    diff,
                                                    last_phase_delta if last_phase_delta is not None else 0,
                                                    okure,
                                                    int(round(delay_seconds))
                                                ])

                                                if control_signal:
                                                    adjust_signal_A_based_on_prediction(pred_t3, pred_t4, record_time, prediction_count)

                                                t3 = pred_t3
                                                t4_actual = count
                                                t4_minus_t3 = t4_actual - t3
                                                latest_data["1"] = (t3, t4_minus_t3)
                                            else:
                                                logger.log_data_incomplete(record_time)
                                                latest_data["1"] = None
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
    queue_delay_file.close()
    phase_log_file.close()
    wasted_green_file.close()
    delay_log_file.close()
    delay_minor_log_file.close()
    green_log_file.close()

    # === CSVファイルを閉じる ===
    for f in queue_delay_files.values():
        f.close()
    total_file.close()



def plot_phase_duration_log(csv_path):
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib
    import datetime

    matplotlib.rcParams['font.family'] = 'Meiryo'

    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        print("⚠️ フェーズ時間ログが空または存在しません。")
        return

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if df.empty or "回数" not in df.columns:
        print("⚠️ データがありません or '回数' 列が存在しません。")
        return

    total_rows = len(df)
    rows_per_hour = total_rows / 24  # ← 小数でもOK

    # 00:00～24:00のラベル（25個）を作成
    tick_labels = [f"{i:02d}:00" for i in range(25)]
    tick_indices = [int(i * rows_per_hour) for i in range(25)]

    # インデックスがデータ数超えないようにフィルタ
    tick_indices = [i for i in tick_indices if i < total_rows]
    tick_labels = tick_labels[:len(tick_indices)]

    plt.figure(figsize=(12, 5))
    plt.plot(df["回数"], df["パターンA(秒)"], label="パターンA（青）", marker="o")

    plt.xlabel("時刻")
    plt.ylabel("フェーズ秒数")
    plt.title("信号A パターンA（青信号）時間の推移")
    plt.xticks(ticks=tick_indices, labels=tick_labels, rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_wasted_green_log(csv_path):
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib
    import datetime

    matplotlib.rcParams['font.family'] = 'Meiryo'

    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        print("⚠️ 無駄青時間ログが空または存在しません。")
        return

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if df.empty or "回数" not in df.columns:
        print("⚠️ データがありません or '回数' 列が存在しません。")
        return

    total_rows = len(df)
    rows_per_hour = total_rows / 24  # ← 小数OKで自動計算

    # 00:00〜24:00 の25本ラベル生成
    tick_labels = [f"{i:02d}:00" for i in range(25)]
    tick_indices = [int(i * rows_per_hour) for i in range(25)]

    # 実データ数を超えないようにフィルタ
    tick_indices = [i for i in tick_indices if i < total_rows]
    tick_labels = tick_labels[:len(tick_indices)]

    plt.figure(figsize=(12, 5))
    plt.plot(df["回数"], df["無駄青時間(s)"], marker='o', label="無駄青時間")

    plt.xlabel("時刻（1時間ごと）")
    plt.ylabel("無駄青時間（秒）")
    plt.title("信号A 無駄青時間（24:00固定）")
    plt.xticks(ticks=tick_indices, labels=tick_labels, rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_combined_phase_and_wasted_log(phase_log_path, wasted_log_path):
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib
    from matplotlib.ticker import MaxNLocator  # ✅ 追加

    matplotlib.rcParams['font.family'] = 'Meiryo'

    # ログファイル読み込み
    if not os.path.exists(phase_log_path) or not os.path.exists(wasted_log_path):
        print("⚠️ ログファイルが存在しません。")
        return

    df_phase = pd.read_csv(phase_log_path, encoding="utf-8-sig")
    df_wasted = pd.read_csv(wasted_log_path, encoding="utf-8-sig")

    if df_phase.empty or df_wasted.empty:
        print("⚠️ ログファイルの中身が空です。")
        return

    total_rows = len(df_phase)
    rows_per_hour = total_rows / 24

    # 時刻ラベル作成
    tick_labels = [f"{i:02d}:00" for i in range(25)]
    tick_indices = [int(i * rows_per_hour) for i in range(25)]
    tick_indices = [i for i in tick_indices if i < total_rows]
    tick_labels = tick_labels[:len(tick_indices)]

    # グラフ描画
    fig, ax1 = plt.subplots(figsize=(12, 5))

    # === 左軸：フェーズ0（パターンA） ===
    ax1.set_xlabel("時刻 (h）")
    ax1.set_ylabel("青時間時間（s）", color="tab:blue")
    ax1.plot(df_phase["回数"], df_phase["パターンA(秒)"], color="tab:blue", marker="o", label="フェーズ0")
    ax1.tick_params(axis='y', labelcolor="tab:blue")
    ax1.set_xticks(tick_indices)
    ax1.set_xticklabels(tick_labels, rotation=45)
    # 自動スケーリング + 自動目盛
    ax1.yaxis.set_major_locator(MaxNLocator(nbins='auto', integer=True))

    # === 右軸：無駄青時間 ===
    ax2 = ax1.twinx()
    ax2.set_ylabel("無駄青時間（s）", color="tab:red")
    ax2.plot(df_wasted["回数"], df_wasted["無駄青時間(s)"], color="tab:red", marker="x", label="無駄青")
    ax2.tick_params(axis='y', labelcolor="tab:red")
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))  # ✅ 整数目盛に固定

    # タイトルとレイアウト
    plt.title("青時間と無駄青時間の推移")
    fig.tight_layout()
    plt.grid(True)
    plt.show()

# === 待ち台数合計 ============
import os
import pandas as pd
from collections import defaultdict
import shutil

def postprocess_logs(log_dir_path):
    # === 各信号機に属する道路ID ===
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

            # 移動と読み込み
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

            # 信号ごとの合計ファイル出力
            combined_df.to_csv(os.path.join(signal_dir, f"信号{signal}_合計.csv"), index=False, encoding="utf-8-sig")
            total_counts_by_signal[signal] = signal_sum
        else:
            total_counts_by_signal[signal] = 0

    # === 全体合計をCSV出力 ===
    all_total = sum(total_counts_by_signal.values())
    total_df = pd.DataFrame([total_counts_by_signal])
    total_df["全体合計"] = all_total
    total_df.to_csv(os.path.join(log_dir_path, "全体合計.csv"), index=False, encoding="utf-8-sig")

    print("✅ ポスト処理完了：信号機別フォルダ生成と全体合計出力")

def plot_green_duration_log(csv_path):
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib
    import os
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

    # === step → 現実時間 (hh:mm形式) に変換 ===
    df["時刻"] = df["step"].apply(lambda s: str(datetime.timedelta(seconds=int(s)))[:-3])

    plt.figure(figsize=(12, 5))

    # === 主道路 ===
    df_main = df[df["種類"] == "主道路"]
    plt.plot(df_main["step"], df_main["青時間(秒)"], marker="o", label="主道路", color="tab:blue")

    # === 従道路 ===
    df_minor = df[df["種類"] == "従道路"]
    plt.plot(df_minor["step"], df_minor["青時間(秒)"], marker="s", label="従道路", color="tab:orange")

    # === X軸設定：1時間ごと（1h=3600秒） ===
    max_step = df["step"].max()
    xticks = np.arange(0, max_step + 3600, 3600)
    xticklabels = [f"{int(h):02d}:00" for h in range(len(xticks))]
    plt.xticks(xticks, xticklabels, rotation=45)

    # === 軸・タイトル ===
    plt.xlabel("現実時間 (hh:mm)")
    plt.ylabel("青信号時間 [秒]")
    plt.title("主道路・従道路の青信号時間の経時変化")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()



# control_signal=False (制御なし), control_signal=True (制御あり)
if __name__ == "__main__":
    run_simulation(sumocfg_path, log_dir_path, control_signal=False)
    postprocess_logs(log_dir_path)
    plot_phase_duration_log(phase_log_path)
    plot_wasted_green_log(wasted_green_log_path)
    plot_combined_phase_and_wasted_log(phase_log_path, wasted_green_log_path)
    plot_green_duration_log(green_log_path)

    plot_time_space_diagram()