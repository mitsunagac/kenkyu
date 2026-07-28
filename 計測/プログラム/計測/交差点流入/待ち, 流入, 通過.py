import traci
import os
import csv
from collections import defaultdict

# ====== 設定 ======
sumocfg_file = r"C:\Users\tslab\Desktop\町モデル\toyama_shouwa.sumocfg"
SUMO_CMD = ["sumo-gui", "-c", sumocfg_file, "--start", "--quit-on-end"]
STEP_LENGTH = 1
INTERVAL = 3600
WARMUP_TIME = 1200
output_dir_base = r"C:\Users\tslab\Desktop\町モデル\output\テスト\待ち台数"
flow_input_dir = r"C:\Users\tslab\Desktop\町モデル\output\テスト\流入"
flow_pass_dir = r"C:\Users\tslab\Desktop\町モデル\output\テスト\通過"

TRAFFIC_LIGHT_CONFIG = {
    "A": [{"id": "1", "edges": ["DtoA"], "red": "yyyyrrrrryyyyrrrrr", "green": "rrrrrrrrrrrrrrrrrr"},],
    "D": [{"id": "2", "edges": ["E91", "E92"], "red": "yyyrrrrryyyrrrrr", "green": "rrrrrrrrrrrrrrrr"},],
    "E": [{"id": "3", "edges": ["E12"], "red": "yyyrrrrrrrr", "green": "rrrrrrrrrrr"},],
    "F": [{"id": "4", "edges":  ["E159"], "red": "yyyrrrrryyyrrrrr", "green": "rrrrrrrrrrrrrrrr"},],
    "I": [{"id": "5", "edges": ["E26"], "red": "rrrryyyrrrryyy", "green": "rrrrrrrrrrrrrr"},],
    "J": [{ "id": "9", "edges": ["M-1toJ"], "red": "yyyyrrrrryyyyrrrrr", "green": "rrrrrrrrrrrrrrrrrr"},
          { "id": "7", "edges": ["-E53"], "red": "rrrryyyyyrrrryyyyy","green": "rrrrrrrrrrrrrrrrrr"},
          { "id": "6", "edges": ["E196"], "red": "rrrryyyyyrrrryyyyy","green": "rrrrrrrrrrrrrrrrrr"},],
    "K": [{"id": "10", "edges": ["E36"], "red": "rrrrrryy", "green": "rrrrrrrr"},],
}

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
        tls_edges_map[tls_id] = sorted(list(incoming_edges))[:4]
    return tls_edges_map

def run_combined_simulation(day, warmup_time=0):
    # 出力フォルダ作成
    wait_dir = os.path.join(output_dir_base, f"day{day}")
    inflow_dir = os.path.join(flow_input_dir, f"day{day}")
    pass_dir = os.path.join(flow_pass_dir, f"day{day}")
    os.makedirs(wait_dir, exist_ok=True)
    os.makedirs(inflow_dir, exist_ok=True)
    os.makedirs(pass_dir, exist_ok=True)

    traci.start(SUMO_CMD)
    print("✅ SUMO開始")

    tls_edges_map = get_tls_incoming_edges_via_traci()

    # 停止車両CSV
    wait_csv_files, wait_writers = {}, {}
    for tl_id, configs in TRAFFIC_LIGHT_CONFIG.items():
        for config in configs:
            path = os.path.join(wait_dir, f"道路{config['id']}_day{day}.csv")
            f = open(path, "w", newline="")
            w = csv.writer(f)
            w.writerow(["Time (s)", "Waiting Cars"])
            wait_csv_files[config['id']] = f
            wait_writers[config['id']] = w

    # 流入CSV
    inflow_writers, inflow_files = {}, {}
    counts_by_tls, seen_vehicle_ids = {}, {}
    for tls_id, edges in tls_edges_map.items():
        path = os.path.join(inflow_dir, f"{tls_id}_inflows.csv")
        f = open(path, "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(f, fieldnames=["time"] + edges)
        writer.writeheader()
        inflow_writers[tls_id] = writer
        inflow_files[tls_id] = f
        counts_by_tls[tls_id] = {edge: 0 for edge in edges}
        seen_vehicle_ids[tls_id] = {edge: set() for edge in edges}

    # 通過車両用CSV
    signal_lanes, signal_passed_vehicles = {}, {}
    for tls_id in traci.trafficlight.getIDList():
        controlled_links = traci.trafficlight.getControlledLinks(tls_id)
        lanes = {link[0] for group in controlled_links for link in group}
        signal_lanes[tls_id] = list(lanes)
        signal_passed_vehicles[tls_id] = set()
        with open(os.path.join(pass_dir, f"{tls_id}.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "vehicle_count"])

    prev_states = {tl_id: ["" for _ in TRAFFIC_LIGHT_CONFIG[tl_id]] for tl_id in TRAFFIC_LIGHT_CONFIG}
    pending_counts = {}

    step = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        sim_time = traci.simulation.getTime()

        for tl_id, configs in TRAFFIC_LIGHT_CONFIG.items():
            for i, config in enumerate(configs):
                current = traci.trafficlight.getRedYellowGreenState(tl_id)
                prev = prev_states[tl_id][i]
                key = (tl_id, i)
                if sim_time >= warmup_time and prev == config['red'] and current == config['green']:
                    pending_counts[key] = sim_time + 2
                if key in pending_counts and sim_time == pending_counts[key]:
                    count = 0
                    for edge in config['edges']:
                        for v in traci.edge.getLastStepVehicleIDs(edge):
                            if traci.vehicle.getSpeed(v) <= 0:
                                count += 1
                    wait_writers[config['id']].writerow([sim_time, count])
                    del pending_counts[key]
                prev_states[tl_id][i] = current

        if sim_time >= warmup_time:
            for tls_id, edges in tls_edges_map.items():
                for edge in edges:
                    for v in traci.edge.getLastStepVehicleIDs(edge):
                        if v not in seen_vehicle_ids[tls_id][edge]:
                            counts_by_tls[tls_id][edge] += 1
                            seen_vehicle_ids[tls_id][edge].add(v)

        if (step - warmup_time) > 0 and (step - warmup_time) % INTERVAL == 0:
            for tls_id, edge_counts in counts_by_tls.items():
                inflow_writers[tls_id].writerow({"time": step, **edge_counts})
                counts_by_tls[tls_id] = {e: 0 for e in edge_counts}
                seen_vehicle_ids[tls_id] = {e: set() for e in edge_counts}

            for tls_id, lanes in signal_lanes.items():
                ids = set()
                for lane in lanes:
                    ids.update(traci.lane.getLastStepVehicleIDs(lane))
                signal_passed_vehicles[tls_id].update(ids)
                with open(os.path.join(pass_dir, f"{tls_id}.csv"), "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([step, len(signal_passed_vehicles[tls_id])])
                signal_passed_vehicles[tls_id] = set()

        step += STEP_LENGTH

    traci.close()
    for f in wait_csv_files.values(): f.close()
    for f in inflow_files.values(): f.close()
    print(f"✅ Day {day}: 全計測完了 → {wait_dir}, {inflow_dir}, {pass_dir}")


if __name__ == "__main__":
    run_combined_simulation(day=1, warmup_time=WARMUP_TIME)
