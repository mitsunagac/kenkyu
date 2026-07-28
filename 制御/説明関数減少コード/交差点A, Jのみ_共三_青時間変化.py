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

# === モデル定義 ===
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
        # print(f"📊 [{self.measurement_count:3d}] 道路{road_id}: t1={t1_value:2d}, t2-t1={t2_minus_t1_value:2d} (step={step})")
    
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
train_csv_path = r"C:\Users\tslab\Desktop\予測\修正済みデータ_5ヶ月分.csv"
adj_path = r"C:\Users\tslab\Desktop\予測\交差点A, Jのみの道路情報.csv"
model_path = r"C:\Users\tslab\Desktop\予測\予測モデル\GCN_epoch_best_model.pth"

# === 使用ノード ===
used_road_indices = [0, 5, 6, 8]                  # 使うノードを4本に絞る
used_road_ids = [str(i+1) for i in used_road_indices]  # ← ['1','6','7','9'] に揃える
latest_data = {rid: None for rid in used_road_ids}     # ← 予測に必要な道路だけ持つ

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
# MAX_RECORD_COUNT = 720 # 記録する最大サイクル数（例：1日分）
MAX_RECORD_COUNT = 300 # 記録する最大サイクル数（例：1日分）
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

# ゴリ押し用現示観測
TRAFFIC_LIGHT_CONFIG = {
    "A": [
    {"id": "1", "edges": ["DtoA"], "red": "rrrryyyyyrrrryyyyy", "green": "rrrrrrrrrrrrrrrrrr"},
    {"id": "10", "edges": ["-E70"], "red": "yyyyrrrrryyyyrrrrr", "green": "rrrrrrrrrrrrrrrrrr"},
    {"id": "11", "edges": ["E2"], "red": "yyyyrrrrryyyyrrrrr", "green": "rrrrrrrrrrrrrrrrrr"},
    {"id": "12", "edges": ["-E1"], "red": "rrrryyyyyrrrryyyyy", "green": "rrrrrrrrrrrrrrrrrr"}
    ],
    "B": [
    {"id": "13", "edges": ["-E5"], "red": "rrrryyyrrrryyy", "green": "rrrrrrrrrrrrrr"},
    {"id": "14", "edges": ["E56"], "red": "yyyyrrryyyyrrr", "green": "rrrrrrrrrrrrrr"},
    {"id": "15", "edges": ["-E2"], "red": "yyyyrrryyyyrrr", "green": "rrrrrrrrrrrrrr"},
    {"id": "16", "edges": ["E55"], "red": "rrrryyyrrrryyy", "green": "rrrrrrrrrrrrrr"}
    ],
    "C": [
    {"id": "17", "edges": ["155393727#0"], "red": "rrrryyyrrrryyy", "green": "rrrrrrrrrrrrrr"},
    {"id": "18", "edges": ["-E77"], "red": "yyyyrrryyyyrrr", "green": "rrrrrrrrrrrrrr"},
    {"id": "19", "edges": ["E76"], "red": "yyyyrrryyyyrrr", "green": "rrrrrrrrrrrrrr"},
    {"id": "20", "edges": ["155390617#0"], "red": "rrrryyyrrrryyy", "green": "rrrrrrrrrrrrrr"}
    ],
    "D": [
    {"id": "2", "edges": ["E91", "E92"], "red": "rrryyyyyrrryyyyy", "green": "rrrrrrrrrrrrrrrr"},
    {"id": "21", "edges": ["-E9"], "red": "yyyrrrrryyyrrrrr", "green": "rrrrrrrrrrrrrrrr"},
    {"id": "22", "edges": ["E4"], "red": "yyyrrrrryyyrrrrr", "green": "rrrrrrrrrrrrrrrr"},
    {"id": "23", "edges": ["-E90"], "red": "rrryyyyyrrryyyyy", "green": "rrrrrrrrrrrrrrrr"},          
    ],
    "E": [
    {"id": "3", "edges": ["E12"], "red": "rrryyyyyyyy", "green": "rrrrrrrrrrr"},
    {"id": "24", "edges": ["-E7"], "red": "yyyrrrrrrrr", "green": "rrrrrrrrrrr"},
    {"id": "25", "edges": ["543210000#5", "-E92"], "red": "rrryyyyyyyy", "green": "rrrrrrrrrrr"},          
    ],
    "F": [
    {"id": "4", "edges": ["E159", "E158"], "red": "rrryyyyyrrryyyyy", "green": "rrrrrrrrrrrrrrrr"},
    {"id": "26", "edges": ["-E123"], "red": "yyyrrrrryyyrrrrr", "green": "rrrrrrrrrrrrrrrr"},
    {"id": "27", "edges": ["E179"], "red": "yyyrrrrryyyrrrrr", "green": "rrrrrrrrrrrrrrrr"},
    {"id": "28", "edges": ["-E12"], "red": "rrryyyyyrrryyyyy", "green": "rrrrrrrrrrrrrrrr"},          
    ],
    "G": [
    {"id": "29", "edges": ["-E190"], "red": "rrrryyyrrrryyy", "green": "rrrrrrrrrrrrrr"},
    {"id": "30", "edges": ["-E186"], "red": "yyyyrrryyyyrrr", "green": "rrrrrrrrrrrrrr"},
    {"id": "31", "edges": ["E185"], "red": "yyyyrrryyyyrrr", "green": "rrrrrrrrrrrrrr"},
    {"id": "32", "edges": ["-E31"], "red": "rrrryyyrrrryyy", "green": "rrrrrrrrrrrrrr"},          
    ],
    "H": [
    {"id": "33", "edges": ["E27"], "red": "rrrryyyrrrryyy", "green": "rrrrrrrrrrrrrr"},
    {"id": "34", "edges": ["-E26"], "red": "yyyyrrryyyyrrr", "green": "rrrrrrrrrrrrrr"},
    {"id": "35", "edges": ["E187", "E186"], "red": "yyyyrrryyyyrrr", "green": "rrrrrrrrrrrrrr"},
    {"id": "36", "edges": ["E25"], "red": "rrrryyyrrrryyy", "green": "rrrrrrrrrrrrrr"},          
    ],
    "I": [
    {"id": "38", "edges": ["-E54"], "red": "rrrryyyrrrryyy", "green": "rrrrrrrrrrrrrr"},
    {"id": "37", "edges": ["-E195"], "red": "yyyyrrryyyyrrr", "green": "rrrrrrrrrrrrrr"},
    {"id": "5", "edges": ["E26"], "red": "yyyyrrryyyyrrr", "green": "rrrrrrrrrrrrrr"},
    {"id": "39", "edges": ["E181"], "red": "rrrryyyrrrryyy", "green": "rrrrrrrrrrrrrr"},          
    ],
    "J": [
    {"id": "9", "edges": ["M-1toJ"], "red": "rrrryyyyyrrrryyyyy", "green": "rrrrrrrrrrrrrrrrrr"},
    {"id": "7", "edges": ["-E53", "-E35"], "red": "yyyyrrrrryyyyrrrrr", "green": "rrrrrrrrrrrrrrrrrr"},
    {"id": "6", "edges": ["E196"], "red": "yyyyrrrrryyyyrrrrr", "green": "rrrrrrrrrrrrrrrrrr"},
    {"id": "40", "edges": ["-E158"], "red": "rrrryyyyyrrrryyyyy", "green": "rrrrrrrrrrrrrrrrrr"}
    ],
    "K": [
    {"id": "41", "edges": ["E34"], "red": "rrrrrryy", "green": "rrrrrrrr"},
    {"id": "8", "edges": ["E36"], "red": "yyyyyyrr", "green": "rrrrrrrr"},
    {"id": "42", "edges": ["E35"], "red": "yyyyyyrr", "green": "rrrrrrrr"}
    ],
    "L": [
    {"id": "43", "edges": ["-E105"], "red": "rrrryyyrrrryyy", "green": "rrrrrrrrrrrrrr"},
    {"id": "44", "edges": ["-E102"], "red": "yyyyrrryyyyrrr", "green": "rrrrrrrrrrrrrr"},
    {"id": "45", "edges": ["E104"], "red": "yyyyrrryyyyrrr", "green": "rrrrrrrrrrrrrr"},
    {"id": "46", "edges": ["E97"], "red": "rrrryyyrrrryyy", "green": "rrrrrrrrrrrrrr"}
    ],
    "M": [
    {"id": "47", "edges": ["-E154"], "red": "rrryyyyyrrryyyyy", "green": "rrrrrrrrrrrrrrrr"},
    {"id": "48", "edges": ["-E218"], "red": "yyyrrrrryyyrrrrr", "green": "rrrrrrrrrrrrrrrr"},
    {"id": "49", "edges": ["E42"], "red": "yyyrrrrryyyrrrrr", "green": "rrrrrrrrrrrrrrrr"},
    {"id": "50", "edges": ["E153"], "red": "rrryyyyyrrryyyyy", "green": "rrrrrrrrrrrrrrrr"}
    ],
    "N": [
    {"id": "51", "edges": ["-E156"], "red": "rrryyyyyrrryyyyy", "green": "rrrrrrrrrrrrrrrr"},
    {"id": "52", "edges": ["-E229"], "red": "yyyrrrrryyyrrrrr", "green": "rrrrrrrrrrrrrrrr"},
    {"id": "53", "edges": ["E259"], "red": "yyyrrrrryyyrrrrr", "green": "rrrrrrrrrrrrrrrr"},
    {"id": "54", "edges": ["E155"], "red": "rrryyyyyrrryyyyy", "green": "rrrrrrrrrrrrrrrr"}
    ],
    "O": [
    {"id": "55", "edges": ["-E0"], "red": "rrryyyyyrrryyyyy", "green": "rrrrrrrrrrrrrrrr"},
    {"id": "56", "edges": ["-E214"], "red": "yyyrrrrryyyrrrrr", "green": "rrrrrrrrrrrrrrrr"},
    {"id": "57", "edges": ["E271"], "red": "yyyrrrrryyyrrrrr", "green": "rrrrrrrrrrrrrrrr"},
    {"id": "58", "edges": ["-E152"], "red": "rrryyyyyrrryyyyy", "green": "rrrrrrrrrrrrrrrr"}
    ],
    "P": [
    {"id": "59", "edges": ["StoP"], "red": "rrryyyyyrrryyyyy", "green": "rrrrrrrrrrrrrrrr"},
    {"id": "60", "edges": ["E80"], "red": "yyyrrrrryyyrrrrr", "green": "rrrrrrrrrrrrrrrr"},
    {"id": "61", "edges": ["E15"], "red": "yyyrrrrryyyrrrrr", "green": "rrrrrrrrrrrrrrrr"},
    {"id": "62", "edges": ["E0"], "red": "rrryyyyyrrryyyyy", "green": "rrrrrrrrrrrrrrrr"}
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

# 初期時間を固定
INITIAL_PHASE_DURATIONS = {
    0: 75,  # フェーズ0：rrrrgGGGgrrrrgGGGg
    3: 33   # フェーズ3：gGGgrrrrrgGGgrrrrr
}

# === 信号制御 ===
def adjust_signal_A_based_on_prediction(t3, t4, step, prediction_count):
    try:
        light_id = "A"
        logic = traci.trafficlight.getAllProgramLogics(light_id)[0]
        old_phases = logic.getPhases()

        if len(old_phases) != 6:
            print(f"⚠️ 信号機Aのフェーズ数が想定と異なります: {len(old_phases)} フェーズ")
            return

        # === 台数に応じた信号時間調整 ===
        # if t3 > t4:
        #     base_delta = 6  # 差があれば最低+8秒
        #     extra_delta = max(0, (t3 - 6) * 2)  # 8台超過分×2秒ずつ加算
        #     total_delta = base_delta + extra_delta

        #     # 最大信号時間 = 91秒 → フェーズ0の最大延長は 91 - 79 = 12秒
        #     delta = min(total_delta, 84 - INITIAL_PHASE_DURATIONS[0])

        if t3 > t4:
            # 1台ごとに2秒延長（6台超過分など関係なく）
            delta = t3 * 2  # 台数差 × 2 秒（上限なし）

        elif t3 < t4:
            delta = t4 * -2
        else:
            delta = 4

        delta = max(-10, min(delta, 10))  # 安全範囲に制限（必要に応じて調整）

        global last_phase_delta, signal_A_adjusted_once
        last_phase_delta = delta
        signal_A_adjusted_once = True  # 初めて制御されたらTrueにする

        from traci._trafficlight import Phase

        new_phases = []
        for i, phase in enumerate(old_phases):
            if i == 0:
                dur = INITIAL_PHASE_DURATIONS[0] + delta
            elif i == 3:
                dur = max(5, INITIAL_PHASE_DURATIONS[3] - delta)
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
                max(5, INITIAL_PHASE_DURATIONS[3] - delta),
                t3,
                t4
            ])

    except Exception as e:
        print(f"⚠️ 信号A制御中エラー: {e}")

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
    t1_time = int(sim_time + base_t1 - delta_adjustment // 2)
    t2_time = int(sim_time + base_t1 + base_t2 - delta_adjustment)
    return t1_time, t2_time


def get_measurement_times_for_minor_road(sim_time, base_t1, base_t2, delta_adjustment=0):
    """
    従道路（道路10・11）の赤信号 前半/後半測定時刻を返す。

    主道路の青信号が延長された場合、従道路の赤信号がその分だけ長くなる。
    t1（前半）は後ろ倒し、t2（後半）も後ろ倒しで調整。

    Parameters:
        sim_time (float): 現在のシミュレーション時刻
        base_t1 (int): 通常の赤信号前半時間（秒）
        base_t2 (int): 通常の赤信号後半時間（秒）
        delta_adjustment (int): 主道路の青信号延長時間（秒）

    Returns:
        (int, int): t1, t2 の測定タイミング
    """
    t1_time = int(sim_time + base_t1 + delta_adjustment // 2)
    t2_time = int(sim_time + base_t1 + base_t2 + delta_adjustment)
    return t1_time, t2_time

last_phase_delta = None  # グローバルに定義（run_simulationの前）
signal_A_adjusted_once = False  # 初期化：制御が1度も入っていない

green_monitoring_active = False
green_monitor_start_time = None
green_monitor_end_time = None
green_monitor_no_pass_seconds = 0
green_monitor_phase_end_time = None

# === DtoA→A→-E1 用 時空間図レコーダ（期間・台数上限つき）===
class TimeSpaceRecorder:
    def __init__(self, edge_from="DtoA", edge_to="-E1",
                 time_range=None,          # 例: (30000, 33600)
                 max_vehicles=None):       # 例: 100
        self.edge_from = edge_from
        self.edge_to   = edge_to
        self.time_range = time_range
        self.max_vehicles = max_vehicles

        # 内部状態
        self.prev_from = set()
        self.prev_to   = set()
        self.enter_from = {}   # veh -> 進入(絶対時刻)
        self.leave_from = {}   # veh -> A到達(絶対時刻)
        self.exit_to    = {}   # veh -> -E1 離脱(絶対時刻)

        self.tracked   = set() # 期間内に採択した車両
        self.completed = set() # -E1 を離脱まで観測できた車両
        self.active = True     # 収集を続けるか（上限達成で False）

    def _accepting(self, sim_time):
        if not self.active:
            return False
        if (self.max_vehicles is not None) and (len(self.tracked) >= self.max_vehicles):
            return False
        if self.time_range is None:
            return True
        t0, t1 = self.time_range
        return (t0 <= sim_time <= t1)

    def step(self, sim_time):
        """シミュレーション毎ステップで呼ぶ"""
        cur_from = set(traci.edge.getLastStepVehicleIDs(self.edge_from))
        cur_to   = set(traci.edge.getLastStepVehicleIDs(self.edge_to))

        # DtoA に存在する車両を、初めて見た瞬間に「採択」判定
        for v in cur_from:
            if v not in self.enter_from and self._accepting(sim_time):
                self.enter_from[v] = sim_time   # 進入時刻（近似）
                self.tracked.add(v)

        # DtoA から消えた瞬間を A 到達の近似時刻とする（tracked のみ記録）
        for v in (self.prev_from - cur_from):
            if v in self.tracked and v not in self.leave_from:
                self.leave_from[v] = sim_time

        # -E1 から消えた瞬間を -E1 離脱時刻とする（tracked のみ記録）
        for v in (self.prev_to - cur_to):
            if v in self.tracked and v not in self.exit_to:
                self.exit_to[v] = sim_time
                self.completed.add(v)
                if self.max_vehicles is not None and len(self.completed) >= self.max_vehicles:
                    self.active = False  # 目的台数に到達したら収集停止

        self.prev_from = cur_from
        self.prev_to   = cur_to

    def build_dataframe(self, len_from, len_to):
        """プロット/CSV用の DataFrame を返す（期間内採択＋完走のみ）"""
        import pandas as pd
        rows = []
        for v in self.completed:
            t0 = self.enter_from.get(v)
            t1 = self.leave_from.get(v)
            t2 = self.exit_to.get(v)
            if t0 is None or t1 is None or t2 is None:
                continue
            if not (t0 <= t1 <= t2):
                continue
            rows.append({
                "veh": v,
                # 絶対時刻
                "enter_abs": t0,
                "A_abs":     t1,
                "E1_abs":    t2,
                # 経過時間（DtoA進入=0）
                "dt_to_A":   t1 - t0,
                "dt_to_E1":  t2 - t0,
                # 距離座標（プロット用）
                "x0": 0.0,
                "x1": float(len_from),
                "x2": float(len_from + len_to),
            })
        return pd.DataFrame(rows).sort_values("enter_abs")

    def plot(self, len_from, len_to, title="DtoA→A→-E1 時空間図"):
        import matplotlib.pyplot as plt
        df = self.build_dataframe(len_from, len_to)
        if df.empty:
            print("⚠️ プロット対象データがありません。")
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        for _, r in df.iterrows():
            ax.plot([r.x0, r.x1, r.x2],
                    [0.0, r.dt_to_A, r.dt_to_E1],
                    alpha=0.8)

        ax.set_xlim(0, len_from + len_to)
        ax.set_xlabel("距離 (m) — DtoA | A交差点 | -E1")
        ax.set_ylabel("経過時間 (s)（DtoA進入時刻を0）")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        # A交差点の縦線
        ax.axvline(len_from, linestyle="--", linewidth=1, alpha=0.6)
        ax.set_xticks([0, len_from, len_from + len_to],
                      labels=["DtoA 入口", "A（交差点）", "-E1 出口"])

        fig.tight_layout()
        plt.show()       # ✅ ウィンドウで出力
        plt.close(fig)   # メモリ解放

    def export_csv(self, out_csv, len_from, len_to):
        import pandas as pd, os
        df = self.build_dataframe(len_from, len_to)
        if df.empty:
            print("⚠️ CSV に書き出すデータがありません。")
            return
        cols = ["veh","enter_abs","A_abs","E1_abs","dt_to_A","dt_to_E1","x0","x1","x2"]
        df.to_csv(out_csv, index=False, encoding="utf-8-sig", columns=cols)
        print(f"✅ 時空間データ CSV を出力しました: {out_csv}")
        return out_csv

    def is_done(self):
        """台数上限に到達、かつ期間の終端も過ぎていれば True（任意で利用）"""
        if self.max_vehicles is None or len(self.completed) < self.max_vehicles:
            return False
        if self.time_range is None:
            return True
        return traci.simulation.getTime() >= self.time_range[1]

# --- Edge 長さ取得ヘルパー（lane 経由。失敗時は sumolib でフォールバック） ---
def get_edge_length(edge_id, sumocfg_path=None, default_len=100.0):
    try:
        n = traci.edge.getLaneNumber(edge_id)  # そのエッジのレーン数
        if n <= 0:
            return default_len
        lengths = []
        for i in range(n):
            lane_id = f"{edge_id}_{i}"  # 通常 edgeID_0, edgeID_1, ...
            try:
                lengths.append(traci.lane.getLength(lane_id))
            except Exception:
                pass
        if lengths:
            return max(lengths)   # ほぼ同じだが最大を採用
    except Exception:
        pass

    # --- フォールバック: net.xml から取得 ---
    if sumocfg_path is not None:
        try:
            import xml.etree.ElementTree as ET, os
            tree = ET.parse(sumocfg_path)
            net_rel = tree.find("./input/net-file").attrib["value"]
            net_path = os.path.join(os.path.dirname(sumocfg_path), net_rel)
            net = sumolib.net.readNet(net_path)
            return float(net.getEdge(edge_id).getLength())
        except Exception as e:
            print(f"⚠️ フォールバック失敗({edge_id}): {e}")

    print(f"⚠️ 長さ取得失敗: {edge_id} → {default_len} m を仮定")
    return float(default_len)



def run_simulation(sumocfg_path, log_dir_path, control_signal=True):
    global prediction_count, last_phase_delta, signal_A_adjusted_once
    global green_monitoring_active, green_monitor_start_time
    global green_monitor_end_time, green_monitor_no_pass_seconds
    global green_monitor_phase_end_time

    if not os.path.exists(log_dir_path):
        os.makedirs(log_dir_path)

    # === スケーラーの読み込み ===
    scaler_path = r"C:\Users\tslab\Desktop\予測\予測モデル\scaler.pkl"
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

    for tl_id, configs in TRAFFIC_LIGHT_CONFIG.items():
        for conf in configs:
            road_id = conf["id"]
            f = open(os.path.join(log_dir_path, f"道路{road_id}.csv"), "w", newline="", encoding="utf-8")
            writer = csv.writer(f)
            writer.writerow(["step", "count"])
            writers[road_id] = writer

    print("🚦 シミュレーション開始...")

    # 時間距離図設定値（可変）=====
    time_range   = (30000, 33600)  # 期間 [s]
    max_vehicles = 100             # 計測する台数

    # 長さ取得（前回パッチの get_edge_length を使用）
    L_DTOA = get_edge_length("DtoA", sumocfg_path=sumocfg_path)
    L_E1   = get_edge_length("-E1", sumocfg_path=sumocfg_path)

    # レコーダ生成（期間＆台数上限を指定）
    ts_rec = TimeSpaceRecorder(edge_from="DtoA", edge_to="-E1",
                            time_range=time_range, max_vehicles=max_vehicles)
    
    # ===========================

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        sim_time = traci.simulation.getTime()

        if ts_rec is not None:
            ts_rec.step(sim_time)

        # === 無駄青監視：全青信号フェーズ末尾の固定秒数を調べる ===
        if green_monitoring_active and green_monitor_start_time <= sim_time < green_monitor_end_time:
            try:
                passing = traci.edge.getLastStepVehicleIDs("DtoA")
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
            print(f"🟩 [青信号全体監視] step={green_monitor_phase_end_time}, "
                f"青信号秒数={monitoring_duration}s, 無駄青={green_monitor_no_pass_seconds}s")
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
                            logic = traci.trafficlight.getAllProgramLogics("A")[0]
                            phase_0_duration = logic.getPhases()[0].duration  # ← 定義通りの青信号秒数（例：79）

                            green_monitor_start_time = int(sim_time)
                            green_monitor_end_time = int(sim_time + phase_0_duration)
                            green_monitor_phase_end_time = green_monitor_end_time
                            green_monitor_no_pass_seconds = 0
                            green_monitoring_active = True

                        # 測定時間設定 (A交差点のみ制御設定あり)
                        if road_id in ["1", "12"]:
                            if road_id in ["1", "12"]:
                                delta = last_phase_delta if signal_A_adjusted_once else 0
                                t1, t2 = get_measurement_times_for_main_road(sim_time, base_t1=21, base_t2=20, delta_adjustment=delta)
                                expected_diff = t2 - t1
                                pending_counts[key] = [
                                    (t1, "t1", road_id, edges, expected_diff),
                                    (t2, "t2", road_id, edges, expected_diff)
                                ]
                        
                        elif road_id in ["10", "11"]:
                            # 予測後に予約された従道路の測定では、予測前に保存されたdeltaを使用
                            delta = last_phase_delta_before_prediction if signal_A_adjusted_once else 0
                            t1, t2 = get_measurement_times_for_minor_road(sim_time, base_t1=42, base_t2=41, delta_adjustment=delta)

                            expected_diff = t2 - t1
                            pending_counts[key].append((t1, "t1", road_id, edges, expected_diff))
                            pending_counts[key].append((t2, "t2", road_id, edges, expected_diff))

                        elif road_id in ["13", "16", "17", "20", "21", "22", "24", "26", "27", "29", "32", "33", "36", "38", "39", "6", "7", "41", "42", "43", "46", "48", "49", "52", "53", "56", "57", "60", "61"]:
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
                                    count = math.ceil(count / 3)
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

                                        if road_id in ["10", "11"]:
                                            delta_for_log = last_phase_delta_before_prediction if control_signal else 0
                                            okure = 84 + delta_for_log
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
                                        
                                        if road_id == "12":
                                            delta_for_log = last_phase_delta_before_prediction if control_signal else 0
                                            okure = 42 - delta_for_log
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

                                        if road_id in ["2", "3","4", "5", "9","14", "15", "18", "19", "23", "25", "28", "30", "31", "34", "35", "37", "40", "44", "45", "47", "50", "51", "54", "55", "58", "59", "62"]:
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
                                        
                                        if road_id in ["13", "16", "17", "20", "21", "22", "24", "26", "27", "29", "32", "33", "36", "38", "39", "6", "7", "8", "41", "42", "43", "46", "48", "49", "52", "53", "56", "57", "60", "61"]:
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

                                                okure = 42 - delta_for_log
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
    phase_log_file.close()
    wasted_green_file.close()
    delay_log_file.close()
    delay_minor_log_file.close()

    return ts_rec   # ← これを追加


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


# control_signal=False (制御なし), control_signal=True (制御あり)
if __name__ == "__main__":

    ts_rec = run_simulation(sumocfg_path, log_dir_path, control_signal=True)

    run_simulation(sumocfg_path, log_dir_path, control_signal=True)
    postprocess_logs(log_dir_path)
    plot_phase_duration_log(phase_log_path)
    plot_wasted_green_log(wasted_green_log_path)
    plot_combined_phase_and_wasted_log(phase_log_path, wasted_green_log_path)

    L_DTOA = get_edge_length("DtoA", sumocfg_path=sumocfg_path)
    L_E1   = get_edge_length("-E1", sumocfg_path=sumocfg_path)

    # 最後にウィンドウで表示
    ts_rec.plot(len_from=L_DTOA, len_to=L_E1)