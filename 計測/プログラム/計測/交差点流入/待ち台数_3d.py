import traci
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
import numpy as np

from matplotlib import rcParams
rcParams['font.family'] = 'MS Gothic'

# ======= 設定 =======
CONFIG_FILE = r"C:\Users\tslab\Desktop\町モデル\toyama_shouwa.sumocfg"  # ←ここは適宜書き換えてください
WAIT_SPEED_THRESHOLD = 0  # 10km/h以下を「待機」とみなす
SUMO_CMD = ["sumo-gui", "-c", CONFIG_FILE, "--start", "--quit-on-end"]

# ======= 信号機構成（J交差点のみ） =======
TRAFFIC_LIGHT_CONFIG = {
    "J": [{ "id": "9", "edges": ["M-1toJ"], "red": "yyyyrrrrryyyyrrrrr", "green": "rrrrrrrrrrrrrrrrrr"},
          { "id": "7", "edges": ["-E53", "-E35"], "red": "rrrryyyyyrrrryyyyy","green": "rrrrrrrrrrrrrrrrrr"},
          { "id": "6", "edges": ["E196"], "red": "rrrryyyyyrrrryyyyy","green": "rrrrrrrrrrrrrrrrrr"},],
}

# TRAFFIC_LIGHT_CONFIG = {
#     "A": [{ "id": "9", "edges": ["DtoA"], "red": "yyyyrrrrryyyyrrrrr", "green": "rrrrrrrrrrrrrrrrrr"},
#           { "id": "7", "edges": ["E2"], "red": "rrrryyyyyrrrryyyyy","green": "rrrrrrrrrrrrrrrrrr"},
#           { "id": "6", "edges": ["-E70"], "red": "rrrryyyyyrrrryyyyy","green": "rrrrrrrrrrrrrrrrrr"},],
# }

# ====== 測定実行（CSV保存なし）======
def run_simulation_and_collect(day=1, warmup_time=0):
    memory_data = defaultdict(list)

    traci.start(SUMO_CMD)
    step = 0
    pending_counts = {}  # {(tls_id, config_index): 予定時刻}
    prev_states = {tl_id: ["" for _ in configs] for tl_id, configs in TRAFFIC_LIGHT_CONFIG.items()}

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        sim_time = traci.simulation.getTime()

        for tl_id, configs in TRAFFIC_LIGHT_CONFIG.items():
            try:
                current_state = traci.trafficlight.getRedYellowGreenState(tl_id)

                for i, config in enumerate(configs):
                    prev = prev_states[tl_id][i]
                    key = (tl_id, i)

                    if sim_time >= warmup_time and prev == config['red'] and current_state == config['green']:
                        pending_counts[key] = sim_time + 2

                    if key in pending_counts and sim_time >= pending_counts[key]:
                        total_count = 0
                        for edge in config['edges']:
                            for veh_id in traci.edge.getLastStepVehicleIDs(edge):
                                if traci.vehicle.getSpeed(veh_id) <= WAIT_SPEED_THRESHOLD:
                                    total_count += 1
                        memory_data[(config['id'], edge)].append((sim_time, total_count))
                        del pending_counts[key]

                    prev_states[tl_id][i] = current_state
            except Exception as e:
                print(f"⚠ Error in TLS {tl_id}, config {config['id']}: {e}")

        step += 1

    traci.close()
    print(f"✅ Day{day} 測定完了（CSV保存なし）")
    return memory_data

# ====== 折れ線3Dグラフ表示 ======S
def plot_3d_trajectory(data):
    # id9_records = data.get(("9", "DtoA"), [])
    id9_records = data.get(("9", "M-1toJ"), [])
    id6_records = []
    id7_records = []

    for (id_, edge), records in data.items():
        if id_ == "6":
            id6_records.extend(records)
        elif id_ == "7":
            id7_records.extend(records)

    num_points = min(len(id9_records), len(id6_records), len(id7_records))
    red_xs, red_ys, red_zs = [], [], []
    blue_xs, blue_ys, blue_zs = [], [], []

    for i in range(num_points):
        _, count_9 = id9_records[i]
        _, count_6 = id6_records[i]
        _, count_7 = id7_records[i]

        sum_67_half = (count_6 + count_7) / 4
        count_9_third = count_9 / 3

        # Z軸 = (サイクル数 * 130) / 1300 = 実時間（時単位）
        time_z = (i * 130) / 3600

        if sum_67_half > count_9_third:
            red_xs.append(sum_67_half)
            red_ys.append(count_9_third)
            red_zs.append(time_z)
        else:
            blue_xs.append(sum_67_half)
            blue_ys.append(count_9_third)
            blue_zs.append(time_z)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 点のみ（線なし）
    ax.plot(blue_xs, blue_ys, blue_zs, marker='o', linestyle='None', color='blue', label='id6+7 ≤ id9')
    ax.plot(red_xs, red_ys, red_zs, marker='o', linestyle='None', color='red', label='id6+7 > id9')

    ax.set_xlabel("従道路の待ち台数")
    ax.set_ylabel("主道路の待ち台数")
    ax.set_zlabel("経過時間（時間単位）")
    ax.set_title("待ち台数推移")

    # ==== Z軸目盛を1ずつにする ====
    max_z = max(red_zs + blue_zs) if (red_zs + blue_zs) else 0
    ax.set_zticks(range(int(max_z) + 2))  # 例: 0〜最大+1 まで1刻み

    plt.tight_layout()
    plt.show()

# ====== 実行部 ======
if __name__ == "__main__":
    data = run_simulation_and_collect(day=1)
    plot_3d_trajectory(data)