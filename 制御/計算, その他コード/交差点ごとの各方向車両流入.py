import traci
import os
import csv
from collections import defaultdict

# ====== 設定 ======
sumocfg_file = r"C:\Users\tslab\Desktop\予測\町モデルデータ\toyama_shouwa.sumocfg"  # ←ファイルパスを調整
SUMO_CMD = ["sumo-gui", "-c", sumocfg_file, "--start", "--quit-on-end"]
STEP_LENGTH = 1
INTERVAL = 3600       # 記録間隔（秒）
WARMUP_TIME = 1200    # 最初の除外時間（秒）
OUTPUT_BASE = r"C:\Users\tslab\Desktop\町モデル\output\inflow_by_tls"
os.makedirs(OUTPUT_BASE, exist_ok=True)

# ====== 信号機ごとの進入エッジを取得 ======
def get_tls_incoming_edges_via_traci():
    tls_edges_map = {}
    for tls_id in traci.trafficlight.getIDList():
        incoming_edges = set()
        controlled_links = traci.trafficlight.getControlledLinks(tls_id)
        for links in controlled_links:
            for conn in links:
                from_lane = conn[0]
                from_edge = traci.lane.getEdgeID(from_lane)
                if not from_edge.startswith(":"):
                    incoming_edges.add(from_edge)
        tls_edges_map[tls_id] = sorted(list(incoming_edges))[:4]  # 最大4方向まで
    return tls_edges_map

# ====== メイン処理 ======
def main():
    traci.start(SUMO_CMD)
    print("SUMOシミュレーション開始")

    tls_edges_map = get_tls_incoming_edges_via_traci()

    # ファイル・カウンタ・車両IDセット初期化
    writers = {}
    files = {}
    counts_by_tls = {}
    seen_vehicle_ids = {}

    for tls_id, edges in tls_edges_map.items():
        file_path = os.path.join(OUTPUT_BASE, f"{tls_id}_inflows.csv")
        f = open(file_path, mode="w", newline="", encoding="utf-8")
        writer = csv.DictWriter(f, fieldnames=["time"] + edges)
        writer.writeheader()
        writers[tls_id] = writer
        files[tls_id] = f
        counts_by_tls[tls_id] = {edge: 0 for edge in edges}
        seen_vehicle_ids[tls_id] = {edge: set() for edge in edges}

    # シミュレーションループ
    step = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()

        # 計測は WARMUP_TIME 以降のみ
        if step >= WARMUP_TIME:
            for tls_id, edges in tls_edges_map.items():
                for edge in edges:
                    vehicles = traci.edge.getLastStepVehicleIDs(edge)
                    for veh_id in vehicles:
                        if veh_id not in seen_vehicle_ids[tls_id][edge]:
                            counts_by_tls[tls_id][edge] += 1
                            seen_vehicle_ids[tls_id][edge].add(veh_id)

            # 一定間隔でCSV出力し、カウントとIDセットをリセット
            if (step - WARMUP_TIME) > 0 and (step - WARMUP_TIME) % INTERVAL == 0:
                for tls_id, edge_counts in counts_by_tls.items():
                    row = {"time": step}
                    row.update(edge_counts)
                    writers[tls_id].writerow(row)
                    counts_by_tls[tls_id] = {edge: 0 for edge in edge_counts}
                    seen_vehicle_ids[tls_id] = {edge: set() for edge in edge_counts}

        step += STEP_LENGTH

    traci.close()
    for f in files.values():
        f.close()
    print("完了：CSVファイルが出力されました")

if __name__ == "__main__":
    main()