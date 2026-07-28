import traci
import csv
import os
from collections import defaultdict

# =============================
# 設定
# =============================
sumo_config = r"C:\Users\tslab\Desktop\町モデル\toyama_shouwa.sumocfg"
# target_edge_ids = ["E24", "-E23"]
target_edge_ids = ["E24"]
interval = 300  # 5分 = 300秒
output_csv = r"C:\Users\tslab\Desktop\町モデル\車線需要\道路1.csv"

os.makedirs(os.path.dirname(output_csv), exist_ok=True)

# =============================
# SUMO開始
# =============================
traci.start(["sumo-gui", "-c", sumo_config, "--start", "--quit-on-end"])

# =============================
# 対象エッジの車線ID取得（安全版）
# =============================
all_lanes = traci.lane.getIDList()
lane_ids = []
for edge in target_edge_ids:
    lane_ids.extend([l for l in all_lanes if l.startswith(edge + "_")])

# 車線ごとの全期間利用車両
lane_total = {lane: set() for lane in lane_ids}

# 車両→使用車線
vehicle_usage = defaultdict(set)

# 5分区間用
interval_lane = {lane: set() for lane in lane_ids}
interval_data = []

current_interval_start = 0

# =============================
# シミュレーション
# =============================
while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()
    current_time = int(traci.simulation.getTime())

    for lane in lane_ids:
        veh_ids = traci.lane.getLastStepVehicleIDs(lane)
        for veh_id in veh_ids:
            lane_total[lane].add(veh_id)
            vehicle_usage[veh_id].add(lane)
            interval_lane[lane].add(veh_id)

    # 5分ごと保存
    if current_time > 0 and current_time % interval == 0:
        interval_data.append(
            (current_interval_start, current_time,
             {lane: len(interval_lane[lane]) for lane in lane_ids})
        )
        interval_lane = {lane: set() for lane in lane_ids}
        current_interval_start = current_time

traci.close()

# =============================
# 車両分類
# =============================
single_lane_count = defaultdict(int)
duplicate_groups = defaultdict(int)

for veh, lanes in vehicle_usage.items():
    if len(lanes) == 1:
        single_lane_count[list(lanes)[0]] += 1
    elif len(lanes) >= 2:
        group_key = tuple(sorted(lanes))
        duplicate_groups[group_key] += 1

# =============================
# CSV出力
# =============================
with open(output_csv, mode="w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)

    # ---- 5分ごと ----
    writer.writerow(["=== 5分ごとの車線利用台数 ==="])
    writer.writerow(["TimeStart", "TimeEnd"] + lane_ids)

    for start, end, data in interval_data:
        row = [start, end]
        for lane in lane_ids:
            row.append(data[lane])
        writer.writerow(row)

    writer.writerow([])

    # ---- 全体合計 ----
    writer.writerow(["=== シミュレーション全体合計 ==="])
    writer.writerow(["Lane", "TotalVehicleCount"])
    for lane in lane_ids:
        writer.writerow([lane, len(lane_total[lane])])

    writer.writerow([])

    # ---- 非重複 ----
    writer.writerow(["=== 単一車線のみ利用車両 ==="])
    writer.writerow(["Lane", "VehicleCount"])
    for lane, count in single_lane_count.items():
        writer.writerow([lane, count])

    writer.writerow([])

    # ---- 重複 ----
    writer.writerow(["=== 重複車線グループ ==="])
    writer.writerow(["LaneGroup", "VehicleCount"])
    for group, count in duplicate_groups.items():
        writer.writerow([" & ".join(group), count])

print("解析完了:", output_csv)