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
train_csv_path = r"C:\Users\tslab\Desktop\予測\修正済みデータ_5ヶ月分.csv"
adj_path = r"C:\Users\tslab\Desktop\予測\富山路線_道路_隣接行列_道路10と道路11を除く.csv"
model_path = r"C:\Users\tslab\Desktop\予測\予測モデル\GCN_epoch_best_model.pth"

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
PREP_TIME = 1300
TRAFFIC_LIGHT_CONFIG = {
    "A": [
    {"id": "1", "edges": ["DtoA"], "red": "rrrryyyyyrrrryyyyy", "green": "rrrrrrrrrrrrrrrrrr"},
    {"id": "10", "edges": ["-E70"], "red": "yyyyrrrrryyyyrrrrr", "green": "rrrrrrrrrrrrrrrrrr"},
    {"id": "11", "edges": ["E2"], "red": "yyyyrrrrryyyyrrrrr", "green": "rrrrrrrrrrrrrrrrrr"}
    ],
    "D": [{"id": "2", "edges": ["E91", "E92"], "red": "rrryyyyyrrryyyyy", "green": "rrrrrrrrrrrrrrrr"}],
    "E": [{"id": "3", "edges": ["E12"], "red": "rrryyyyyyyy", "green": "rrrrrrrrrrr"}],
    "F": [{"id": "4", "edges": ["E159", "E158"], "red": "rrryyyyyrrryyyyy", "green": "rrrrrrrrrrrrrrrr"}],
    "I": [{"id": "5", "edges": ["E26"], "red": "rrrryyyrrrryyy", "green": "rrrrrrrrrrrrrr"}],
    "J": [
        {"id": "9", "edges": ["M-1toJ"], "red": "rrrryyyyyrrrryyyyy", "green": "rrrrrrrrrrrrrrrrrr"},
        {"id": "7", "edges": ["-E53", "-E35"], "red": "yyyyrrrrryyyyrrrrr", "green": "rrrrrrrrrrrrrrrrrr"},
        {"id": "6", "edges": ["E196"], "red": "yyyyrrrrryyyyrrrrr", "green": "rrrrrrrrrrrrrrrrrr"}
    ],
    "K": [{"id": "8", "edges": ["E36"], "red": "yyyyyyrr", "green": "rrrrrrrr"}],
}

prediction_count = 0

prediction_log_path = r"C:\Users\tslab\Desktop\予測\制御\csv掃き出し\予測結果_道路1.csv"
prediction_log_file = open(prediction_log_path, "w", newline="", encoding="utf-8-sig")
prediction_writer = csv.writer(prediction_log_file)
prediction_writer.writerow(["回数", "step", "前半", "後半", "予測前半", "予測後半"])

# === フェーズ時間ログファイル ===
phase_log_path = r"C:\Users\tslab\Desktop\予測\制御\csv掃き出し\信号A_フェーズ時間ログ.csv"
phase_log_file = open(phase_log_path, "w", newline="", encoding="utf-8-sig")
phase_writer = csv.writer(phase_log_file)
phase_writer.writerow(["回数", "step", "パターンA(秒)", "パターンB(秒)", "予測t3", "予測t4"])


# 初期時間を固定
INITIAL_PHASE_DURATIONS = {
    0: 79,  # フェーズ0：rrrrgGGGgrrrrgGGGg
    3: 39   # フェーズ3：gGGgrrrrrgGGgrrrrr
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
        if t3 > t4:
            base_delta = 6  # 差があれば最低+8秒
            extra_delta = max(0, (t3 - 6) * 2)  # 8台超過分×2秒ずつ加算
            total_delta = base_delta + extra_delta

            # 最大信号時間 = 91秒 → フェーズ0の最大延長は 91 - 79 = 12秒
            delta = min(total_delta, 91 - INITIAL_PHASE_DURATIONS[0])
        elif t3 < t4:
            delta = -4
        else:
            delta = 4

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

def run_simulation(sumocfg_path, log_dir_path):
    if not os.path.exists(log_dir_path):
        os.makedirs(log_dir_path)

    # === スケーラーの読み込み ===
    scaler_path = r"C:\Users\tslab\Desktop\予測\予測モデル\scaler.pkl"
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"❌ スケーラーファイルが存在しません: {scaler_path}")

    scaler = joblib.load(scaler_path)
    print("✅ スケーラー読み込み完了")

    sumoBinary = sumolib.checkBinary('sumo-gui')
    traci.start([sumoBinary, "-c", sumocfg_path])
    sim_time = 0
    prev_states = defaultdict(dict)
    pending_counts = defaultdict(list)
    first_half_storage = defaultdict(dict)

    writers = {}
    scheduled_times_A = set()

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

        for tl_id, configs in TRAFFIC_LIGHT_CONFIG.items():
            for i, config in enumerate(configs):
                try:
                    key = (tl_id, i)
                    current = traci.trafficlight.getRedYellowGreenState(tl_id)
                    prev = prev_states[tl_id].get(i, "")

                    if sim_time >= PREP_TIME and prev == config["red"] and current == config["green"]:
                        road_id = config["id"]
                        edges = config["edges"]

                        if road_id == "1":
                            if road_id == "1":
                                delta = last_phase_delta if signal_A_adjusted_once else 0
                                t1, t2 = get_measurement_times_for_main_road(sim_time, base_t1=24, base_t2=23, delta_adjustment=delta)
                                expected_diff = t2 - t1
                                pending_counts[key] = [
                                    (t1, "t1", road_id, edges, expected_diff),
                                    (t2, "t2", road_id, edges, expected_diff)
                                ]
                        
                        elif road_id in ["10", "11"]:
                            delta = last_phase_delta if signal_A_adjusted_once else 0
                            t1, t2 = get_measurement_times_for_minor_road(sim_time, base_t1=43, base_t2=43, delta_adjustment=delta)
                            expected_diff = t2 - t1
                            pending_counts[key].append((t1, "t1", road_id, edges, expected_diff))
                            pending_counts[key].append((t2, "t2", road_id, edges, expected_diff))

                        else:
                            t1 = int(sim_time + 24)
                            t2 = int(sim_time + 47)
                            pending_counts[key].append((t1, "t1", road_id, edges, 23))
                            pending_counts[key].append((t2, "t2", road_id, edges, 23))

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
                                if id_int in [1, 2, 3, 4, 9]:
                                    count = math.ceil(count / 3)
                                elif id_int in [5, 6, 7, 8, 10, 11]:
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

                                        writers[road_id].writerow([record_time, diff])
                                        del first_half_storage[road_id][t1_time]
                                        latest_data[road_id] = (prev_count, diff)
                                        logger.log_measurement_data(road_id, prev_count, diff, record_time)

                                        if road_id == "1":
                                            global prediction_count
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
                                                prediction_writer.writerow([prediction_count, record_time, t1_count, diff, pred_t3, pred_t4])

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


def plot_phase_duration_log(csv_path):
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.rcParams['font.family'] = 'Meiryo'  # 日本語対応フォント（例: Windows）

    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        print("⚠️ フェーズ時間ログが空または存在しません。")
        return

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if df.empty:
        print("⚠️ CSVは存在しますがデータがありません。")
        return

    # パターンA（フェーズ0）のみを描画
    plt.figure(figsize=(10, 5))
    plt.plot(df["回数"], df["パターンA(秒)"], label="パターンA（青）", marker="o")

    plt.xlabel("予測回数")
    plt.ylabel("フェーズ秒数")
    plt.title("信号A パターンA（青信号）時間の推移")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_simulation(sumocfg_path, log_dir_path)
    plot_phase_duration_log(phase_log_path)