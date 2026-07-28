import os
import traci
import sumolib
import csv
import math
from collections import defaultdict

# === 設定 ===
sumocfg_path = r"C:\Users\tslab\Desktop\予測\町モデルデータ\toyama_shouwa.sumocfg"
log_dir_path = r"C:\Users\tslab\Desktop\予測\制御\csv掃き出し\制御前_各道路計測ログ"


# === 設定 ===
PREP_TIME = 1000
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
RED_TO_GREEN = {"r", "y"}

def run_simulation(sumocfg_path, log_dir_path):
    if not os.path.exists(log_dir_path):
        os.makedirs(log_dir_path)

    sumoBinary = sumolib.checkBinary('sumo-gui')
    traci.start([sumoBinary, "-c", sumocfg_path])

    sim_time = 0
    prev_states = defaultdict(dict)
    pending_counts = defaultdict(list)  # key=(tl_id, index) → list of (record_time, tag, road_id, edges)
    first_half_storage = defaultdict(dict)  # {road_id: {record_time_t1: count}}

    writers = {}
    for tl_id, configs in TRAFFIC_LIGHT_CONFIG.items():
        for conf in configs:
            road_id = conf["id"]
            f = open(os.path.join(log_dir_path, f"道路{road_id}.csv"), "w", newline="", encoding="utf-8")
            writer = csv.writer(f)
            writer.writerow(["step", "count"])
            writers[road_id] = writer

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        sim_time = traci.simulation.getTime()

        for tl_id, configs in TRAFFIC_LIGHT_CONFIG.items():
            for i, config in enumerate(configs):
                try:
                    key = (tl_id, i)
                    current = traci.trafficlight.getRedYellowGreenState(tl_id)
                    prev = prev_states[tl_id].get(i, "")

                    # 赤→緑の変化検出
                    if sim_time >= PREP_TIME and prev == config["red"] and current == config["green"]:
                        road_id = config["id"]
                        edges = config["edges"]
                        if road_id in ["10", "11"]:
                            # 特別なタイミング
                            t1 = int(sim_time + 44)
                            t2 = int(sim_time + 87)
                        else:
                            # 通常のタイミング
                            t1 = int(sim_time + 24)
                            t2 = int(sim_time + 47)
                        pending_counts[key].append((t1, "t1", road_id, edges))
                        pending_counts[key].append((t2, "t2", road_id, edges))

                    # 計測
                    new_pending = []
                    for record_time, tag, road_id, edges in pending_counts[key]:
                        record_time = int(record_time)
                        if sim_time >= record_time:
                            count = 0
                            for edge in edges:
                                for veh_id in traci.edge.getLastStepVehicleIDs(edge):
                                    if traci.vehicle.getSpeed(veh_id) <= 0:
                                        count += 1

                            # 割り算処理（切り上げ）
                            try:
                                id_int = int(road_id)
                                if id_int in [1, 2, 3, 4, 9]:
                                    count = math.ceil(count / 3)
                                elif id_int in [5, 6, 7, 8, 10, 11]:
                                    count = math.ceil(count / 2)
                            except:
                                pass  # IDが数値でない場合は割らずにそのまま

                            if tag == "t1":
                                record_time = int(record_time)
                                first_half_storage[road_id][record_time] = count
                                writers[road_id].writerow([record_time, count])
                                # print(f"✅ 計測@24s 道路{road_id} → step={record_time}, 台数={count}")

                            elif tag == "t2":
                                record_time = int(record_time)
                                # 差分記録時
                                if road_id in ["10", "11"]:
                                    time_offset = 43  # t2 - t1 for road 10, 11
                                else:
                                    time_offset = 23  # default offset

                                for delta in [0, -1, 1]:
                                    t1_time = record_time - time_offset + delta
                                    if t1_time in first_half_storage[road_id]:
                                        prev_count = first_half_storage[road_id][t1_time]
                                        diff = count - prev_count
                                        writers[road_id].writerow([record_time, diff])
                                        # print(f"✅ 計測@{record_time}s 道路{road_id} → 差分={diff}（t1={t1_time}）")
                                        del first_half_storage[road_id][t1_time]
                                        break
                                else:
                                    print(f"⚠ 計測@{record_time}s スキップ：道路{road_id} の ±1秒範囲に t1 が見つかりません")

                        else:
                            new_pending.append((record_time, tag, road_id, edges))

                    pending_counts[key] = new_pending
                    prev_states[tl_id][i] = current

                except Exception as e:
                    print(f"⚠ Error at {tl_id} ({config['id']}) - {e}")

    traci.close()


if __name__ == "__main__":
    run_simulation(sumocfg_path, log_dir_path)