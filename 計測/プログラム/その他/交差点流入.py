import traci
import os
import csv
from collections import defaultdict

# ======= 設定 =======
sumocfg_file = r"C:\Users\tslab\Desktop\町モデル\toyama_shouwa.sumocfg"  # ←あなたのSUMO設定ファイルに置き換えてください
SUMO_CMD = ["sumo-gui", "-c", sumocfg_file, "--start", "--quit-on-end"]
STEP_LENGTH = 1
INTERVAL = 300  # 1時間ごと
WARMUP_TIME = 1200  # ウォームアップ期間（あれば指定）

# 出力先フォルダ（交差点ごと）
OUTPUT_DIR = "output/交差点流入"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 車両ごとの前回レーンを記録
vehicle_last_lane = {}

def get_tls_fromLanes_map():
    """
    信号機ごとに、流入レーン集合（fromLane → tls_id）をマッピング
    """
    fromLane_to_tls = {}
    for tls_id in traci.trafficlight.getIDList():
        links = traci.trafficlight.getControlledLinks(tls_id)
        for link_list in links:
            for link in link_list:
                fromLane = link[0]
                fromEdge = traci.lane.getEdgeID(fromLane)
                if not fromEdge.startswith(":"):  # 内部エッジ除外
                    fromLane_to_tls[fromLane] = tls_id
    return fromLane_to_tls

def write_final_counts(tls_hourly_counts):
    for tls_id, hourly_list in tls_hourly_counts.items():
        all_from_edges = set()
        for count_dict in hourly_list:
            all_from_edges.update(count_dict.keys())
        all_from_edges = sorted(all_from_edges)

        filename = os.path.join(OUTPUT_DIR, f"{tls_id}.csv")
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time"] + all_from_edges)
            for i, count_dict in enumerate(hourly_list):
                time = (i+1) * INTERVAL
                row = [time] + [count_dict.get(edge, 0) for edge in all_from_edges]
                writer.writerow(row)
        print(f"✅ {filename} を出力しました")

def main():
    traci.start(SUMO_CMD)
    step = 0
    tls_fromLane_map = get_tls_fromLanes_map()

    # tls_id → 時間ごとのリスト → fromEdgeごとのカウント
    tls_hourly_counts = defaultdict(list)
    current_counts = defaultdict(lambda: defaultdict(int))  # tls_id → fromEdge → count

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        step += STEP_LENGTH

        # すべての車両についてレーンの移動を確認
        for vid in traci.vehicle.getIDList():
            current_lane = traci.vehicle.getLaneID(vid)
            if vid in vehicle_last_lane:
                prev_lane = vehicle_last_lane[vid]
                if prev_lane != current_lane:
                    # 交差点通過があったか確認
                    if prev_lane in tls_fromLane_map:
                        tls_id = tls_fromLane_map[prev_lane]
                        from_edge = traci.lane.getEdgeID(prev_lane)
                        current_counts[tls_id][from_edge] += 1
            vehicle_last_lane[vid] = current_lane

        # 1時間ごとに記録
        if step % INTERVAL == 0:
            for tls_id in current_counts:
                tls_hourly_counts[tls_id].append(dict(current_counts[tls_id]))
                current_counts[tls_id].clear()
            # print(f"⏱️ {step} 秒：交差点への流入台数を記録しました")

    traci.close()
    write_final_counts(tls_hourly_counts)

if __name__ == "__main__":
    main()