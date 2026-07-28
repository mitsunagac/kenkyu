import random
import csv
import sumolib
import traci
import os
import xml.etree.ElementTree as ET

# ファイル統合用
import pandas as pd
import glob

# ファイル移動用
import shutil

# 車両生成用
from collections import deque

# ======================
# 車両生成設定
# ======================
netfile = r"C:\Users\tslab\Desktop\町モデル\toyama_shouwa_offset_smz.net.xml" #オフセット設定あり
# netfile = r"C:\Users\tslab\Desktop\町モデル\toyama_shouwa_no_offset.net.xml" #オフセット設定なし
to_edge_csv = r"C:\Users\tslab\Desktop\町モデル\output\交通環境調査\edges_with_2_or_more_lanes2.csv"
highway_edge_csv = to_edge_csv
rou_outputfile = r"C:\Users\tslab\Desktop\町モデル\random_trips_with_highway.rou.xml"
generation_log_base = r"C:\Users\tslab\Desktop\町モデル\output\発生台数\車両発生記録_day{}.csv"
kakuritsu_log_base = r"C:\Users\tslab\Desktop\町モデル\output\確率変化地点\確率変化地点_day{}.csv"

other_edge_csv = r"C:\Users\tslab\Desktop\町モデル\output\交通環境調査\edges_with_1_lane2.csv"

# ===== 朝・昼・夜 各時間帯の出発地エリア =====
# グループ定義（※from_edge 選定のために1回だけ定義すればOK）
group1_1 = ['StoP']
group1_2 = ['-E37', 'E184']
group2_1 = ['-E1']
group2_2 = ['E52', '-E48']
used_edges = set(group1_1 + group1_2 + group2_1 + group2_2)

# 時間帯ごとの第1層の重み（南・北・その他）
top_level_weights_by_time = [
    [0.5, 0.4, 0.2],  # 深夜
    [0.5, 0.4, 0.1],  # 早朝
    [0.6, 0.3, 0.1],  # 朝 (ピーク)
    [0.5, 0.4, 0.1],  # 昼
    [0.4, 0.5, 0.1],  # 昼過ぎ
    [0.3, 0.6, 0.1],  # 夕方 (ピーク)
    [0.4, 0.5, 0.1],  # 夜
]

# 第2層サブグループの重み
group1_weights = [0.7, 0.3]  # 南内の比率
group2_weights = [0.7, 0.3]  # 北内の比率

# ===== 朝・昼・夜 各時間帯の目的地地エリア =====
# 目的地グループをループ外に定義
to_group1 = ['E1', '-E52', 'E48']
to_group2 = ['-E14', 'E37', '-E184']


# ===== 時間帯を朝昼夜のインデックスにマッピング =====
def get_time_index(period_index):
    if period_index == 0:
        return 0  # 準備時間帯は早朝扱い
    elif 1 <= period_index <= 4:
        return 0  # 深夜 1~4
    elif 4 <= period_index <= 7:
        return 1  # 早朝 4~7
    elif 7 <= period_index <= 10:
        return 2  # 朝(ピーク) 7~10
    elif 10 <= period_index <= 13:
        return 3  # 昼 10~13
    elif 13 <= period_index <= 17:
        return 4  # 昼過ぎ 13~17
    elif 17 <= period_index <= 20:
        return 5  # 夕方(ピーク) 17~120
    else:
        return 6  # 夜 20~24

# ======================
# 車両生成設定　(テスト用)
# ====================== 
# ===== 時間帯を朝昼夜のインデックスにマッピング =====
# def get_time_index(period_index):
#     if period_index == 0:
#         return 0  # 準備時間帯は早朝扱い
#     elif 1 <= period_index <= 2:
#         return 0  # 早朝 0
#     elif 2 <= period_index <= 3:
#         return 1  # 朝(ピーク) 1
#     elif 3 <= period_index <= 4:
#         return 2  # 昼 2
#     elif 4 <= period_index <= 5:
#         return 3  # 昼過ぎ 3
#     elif 5 <= period_index <= 6:
#         return 4  # 夕方(ピーク) 4
#     else:
#         return 5  # 夜 5

# ======================
# 複数信号監視設定（エッジ単位で保存）
# ======================
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

# 逆方向
# TRAFFIC_LIGHT_CONFIG = {
#     "A": [{"id": "1", "edges": ["-E1"], "red": "yyyyrrrrryyyyrrrrr", "green": "rrrrrrrrrrrrrrrrrr"},],
#     "D": [{"id": "2", "edges": ["-E90"], "red": "yyyrrrrryyyrrrrr", "green": "rrrrrrrrrrrrrrrr"},],
#     "E": [{"id": "3", "edges": ["543210000#5", "-E92"], "red": "yyyrrrrrrrr", "green": "rrrrrrrrrrr"},],
#     "F": [{"id": "4", "edges":  ["-E12"], "red": "yyyrrrrryyyrrrrr", "green": "rrrrrrrrrrrrrrrr"},],
#     "J": [{ "id": "9", "edges": ["-E158"], "red": "yyyyrrrrryyyyrrrrr", "green": "rrrrrrrrrrrrrrrrrr"},],
# }

output_dir_base = r"C:\Users\tslab\Desktop\町モデル\output\待ち台数"
sumocfg_file = r"C:\Users\tslab\Desktop\町モデル\toyama_shouwa.sumocfg"
SUMO_CMD = ["sumo-gui", "-c", sumocfg_file, "--start", "--quit-on-end"]


# ======================
# 流入量測定用設定
# ======================
flow_pass_dir = r"C:\Users\tslab\Desktop\町モデル\output\車両流入"

step_length = 1           # 1ステップ = 1秒
interval = 3600            # 記録間隔（秒）

# ======================
# 通過量測定用設定
# ======================
flow_input_dir = r"C:\Users\tslab\Desktop\町モデル\output\通過"

# ======================
# ファイル統合設定
# ======================
machi_folder = r"C:\Users\tslab\Desktop\町モデル\output\待ち台数"
output_filename = "all_days_待ち_統合.csv"

input_base_folder_ryunyu = r"C:\Users\tslab\Desktop\町モデル\output\車両流入"
output_folder_ryunyu = r"C:\Users\tslab\Desktop\町モデル\output\車両流入\統合"

# ======================
# ファイル移動設定
# ======================
source = r"C:\Users\tslab\Desktop\町モデル\random_trips_with_highway.rou.xml"
destination = r"C:\Users\tslab\Desktop\町モデル\output\生成された車"

# ======================
# 車両生成関数
# ======================
# ===== 幹線内だけで到達可能な経路を探索するBFS関数 =====
def find_path_on_highways(from_edge_id, to_edge_id, net, highway_edges):
    visited = set()
    queue = deque([[from_edge_id]])

    while queue:
        path = queue.popleft()
        current_edge_id = path[-1]

        if current_edge_id == to_edge_id:
            return [net.getEdge(e) for e in path]

        if current_edge_id in visited:
            continue
        visited.add(current_edge_id)

        try:
            current_edge = net.getEdge(current_edge_id)
            for succ_edge in current_edge.getOutgoing():
                succ_id = succ_edge.getID()
                if succ_id in highway_edges and succ_id not in visited:
                    queue.append(path + [succ_id])
        except:
            continue

    return None

# ===== 出発地・目的地に応じたルート生成 =====
def generate_vehicle_routes(all_time_settings, day):
    generation_log_csv = generation_log_base.format(day)
    os.makedirs(os.path.dirname(generation_log_csv), exist_ok=True)

    kakuritsu_log_csv = kakuritsu_log_base.format(day)
    os.makedirs(os.path.dirname(kakuritsu_log_csv), exist_ok=True)

    net = sumolib.net.readNet(netfile)
    all_edges = [e.getID() for e in net.getEdges() if not e.getID().startswith(":")]

    all_to_edges = []
    with open(to_edge_csv, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            edge_id = row['edge_id'].strip().strip("'\"")
            if edge_id:
                all_to_edges.append(edge_id)

    highway_edges = []
    with open(highway_edge_csv, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            edge_id = row['edge_id'].strip().strip("'\"")
            if edge_id:
                highway_edges.append(edge_id)

    otherway_edges = []
    with open(other_edge_csv, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            edge_id = row['edge_id'].strip().strip("'\"")
            if edge_id:
                otherway_edges.append(edge_id)

    reachable_edges_from_highways = set()
    for hwy_id in highway_edges:
        try:
            hwy_edge = net.getEdge(hwy_id)
            for edge in all_to_edges:
                try:
                    edge_obj = net.getEdge(edge)
                    path, _ = net.getShortestPath(hwy_edge, edge_obj)
                    if path:
                        reachable_edges_from_highways.add(edge)
                except:
                    continue
        except:
            continue

    to_group3 = [e for e in highway_edges if e not in to_group1 and e not in to_group2]

    def pick_destination(from_edge, selected_top_group, is_from_highway, best_hwy, net, highway_edges, reachable_edges_from_highways):
        if selected_top_group == 0:
            candidates = to_group1
        elif selected_top_group == 1:
            candidates = to_group2
        else:
            candidates = to_group3

        if not is_from_highway and best_hwy:
            reachable_targets = [
                e for e in candidates
                if e != from_edge and find_path_on_highways(best_hwy, e, net, set(highway_edges))
            ]
            if not reachable_targets:
                return None
            return random.choice(reachable_targets)

        filtered = [e for e in candidates if e in reachable_edges_from_highways and e != from_edge]
        if not filtered:
            return None
        return random.choice(filtered)

    valid_routes = []
    vehicle_index = 0
    current_time = 0
    vehicle_counts = []

    # カウント初期化
    maji_count = 0
    maji_count_1 = 0
    maji_count_2 = 0
    maji_count_3 = 0
    maji_count_4 = 0
    maji_count_5 = 0
    all_maji_count = 0

    for period_index, (sim_duration, depart_min, depart_max) in enumerate(all_time_settings):
        time_limit = current_time + sim_duration
        count_this_period = 0
        time_index = get_time_index(period_index)
        top_weights = top_level_weights_by_time[time_index]

        group3 = [e for e in otherway_edges]
        from_area_choices = [group1_1 + group1_2, group2_1 + group2_2, group3]

        while current_time <= time_limit:
            top_group_index = random.choices([0, 1, 2], weights=top_weights, k=1)[0]

            if top_group_index == 0:
                sub_group = random.choices([group1_1, group1_2], weights=group1_weights, k=1)[0]
            elif top_group_index == 1:
                sub_group = random.choices([group2_1, group2_2], weights=group2_weights, k=1)[0]
            else:
                sub_group = group3

            from_edge = random.choice(sub_group)
            is_from_highway = from_edge in highway_edges

            if is_from_highway:
                to_edge = pick_destination(from_edge, top_group_index, True, None, net, highway_edges, reachable_edges_from_highways)
                if not to_edge or from_edge == to_edge:
                    # print(f"[⚠️目的地エラー] from: {from_edge}, to: {to_edge} （目的地が無効か同一）")
                    continue
                path_on_highway = find_path_on_highways(from_edge, to_edge, net, set(highway_edges))
                if not path_on_highway:
                    # print(f"[⚠️BFS失敗] 幹線: {from_edge} → to: {to_edge} への経路が見つかりません")
                    continue
                final_route = path_on_highway
            else:
                from_edge_obj = net.getEdge(from_edge)
                min_dist = float('inf')
                best_hwy = None
                path_to_highway = []

                for hwy_edge in highway_edges:
                    try:
                        hwy_edge_obj = net.getEdge(hwy_edge)
                        path, dist = net.getShortestPath(from_edge_obj, hwy_edge_obj)
                        if path and dist < min_dist:
                            best_hwy = hwy_edge
                            path_to_highway = path
                            min_dist = dist
                    except:
                        continue

                if not path_to_highway or not best_hwy:
                    # print(f"[⚠️接続失敗] from: {from_edge} → 幹線に接続できません")
                    continue

                to_edge = pick_destination(from_edge, top_group_index, False, best_hwy, net, highway_edges, reachable_edges_from_highways)
                if not to_edge or from_edge == to_edge:
                    # print(f"[⚠️目的地エラー] from: {from_edge}, to: {to_edge} （目的地が無効か同一）")
                    continue

                path_to_dest = find_path_on_highways(best_hwy, to_edge, net, set(highway_edges))
                if not path_to_dest:
                    # print(f"[⚠️BFS失敗] 幹線: {best_hwy} → to: {to_edge} への経路が見つかりません")
                    continue

                if path_to_highway[-1].getID() == path_to_dest[0].getID():
                    path_to_dest = path_to_dest[1:]

                final_route = list(path_to_highway) + path_to_dest
                # print(f"[✅幹線経由] from: {from_edge} → 幹線: {best_hwy} → to: {to_edge}")

            route_ids = [e.getID() for e in final_route]
            valid_routes.append((vehicle_index, current_time, route_ids))
            print(f"⏱ 車両{vehicle_index} を {current_time} 秒目に生成中...")


            # 特定のエッジからの出発をカウント (朝)
            if from_edge == "StoP": 
                maji_count += 1 
                # print("🎯 出発地に変えた場所が選ばれました")
            if from_edge == "-E37": 
                maji_count_1 += 1
                # print("🎯 出発地に変えた場所が選ばれました")
            if from_edge == "E184": 
                maji_count_2 += 1
                # print("🎯 出発地に変えた場所が選ばれました")

            # 特定のエッジからの出発をカウント (夜)
            if from_edge == "-E1": 
                maji_count_3 += 1
                # print("🔹 出発地に変えた場所が選ばれました")
            if from_edge == "E52": 
                maji_count_4 += 1
                # print("🔹 出発地に変えた場所が選ばれました")
            if from_edge == "-E48": 
                maji_count_5 += 1
                # print("🔹 出発地に変えた場所が選ばれました")

            vehicle_index += 1
            current_time += random.randint(depart_min, depart_max)
            count_this_period += 1

        vehicle_counts.append((period_index + 1, count_this_period))


    # rou.xml出力
    with open(rou_outputfile, "w") as f:
        f.write("<routes>\n")
        f.write('  <vType id="car" accel="2.5" decel="4.5" maxSpeed="16.67" length="5" sigma="0.5"/>\n')
        for veh_id, depart_time, route_edges in valid_routes:
            route_str = ' '.join(route_edges)
            f.write(f'  <route id="route{veh_id}" edges="{route_str}"/>\n')
            f.write(f'  <vehicle id="veh{veh_id}" type="car" route="route{veh_id}" depart="{depart_time}"/>\n')
        f.write("</routes>\n")

    # ログCSV出力
    with open(generation_log_csv, mode='w', newline='') as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["時間帯", "生成台数"])
        for period, count in vehicle_counts:
            writer.writerow([f"時間帯{period}", count])

    # ログcsv出力 (確率変化地点)
    with open(kakuritsu_log_csv, mode='w', newline='') as log_file:
        writer = csv.writer(log_file)
        all_maji_count = maji_count + maji_count_1 + maji_count_2 + maji_count_3 + maji_count_4 + maji_count_5
        writer.writerow(["合計発生台数", "南", "東", "西", "北", "北西", "北東"])
        writer.writerow([all_maji_count, maji_count, maji_count_1, maji_count_2, maji_count_3, maji_count_4, maji_count_5])

    # 特定出発地からのカウントを表示
    all_maji_count = maji_count + maji_count_1 + maji_count_2 + maji_count_3 + maji_count_4 + maji_count_5
    print(f"✅ Day {day}: {vehicle_index} 台の車両を生成しました: {rou_outputfile}")
    print(f"🎯 南から{maji_count} 台, 東から{maji_count_1} 台, 西から{maji_count_2} 台")
    print(f"🎯 北から{maji_count_3} 台, 北西から{maji_count_4} 台, 北東から{maji_count_5} 台")
    print(f"✅ 確率変更点から{all_maji_count} 台, 生成されました")


# ======================
# 信号機ごとの各方向エッジごとに計測 & 流入量測定
# ======================
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

def run_simulation_with_signal_logging(day, warmup_time=0):
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

        if (step - warmup_time) > 0 and (step - warmup_time) % interval == 0:
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

        step += step_length

    traci.close()
    for f in wait_csv_files.values(): f.close()
    for f in inflow_files.values(): f.close()
    print(f"✅ Day {day}: 全計測完了 → {wait_dir}, {inflow_dir}, {pass_dir}")


# ======================
# ファイル統合(待ち台数)
# ======================
def combine_csv_vertically(input_base_folder, output_filename, days):
    # 出力先のフォルダ「統合」を作成（存在しない場合のみ）
    output_folder = os.path.join(input_base_folder, "統合")
    os.makedirs(output_folder, exist_ok=True)

    # 結合後のCSVファイルの保存パスを作成
    final_output_path = os.path.join(output_folder, output_filename)

    # 全日分のデータフレームを格納するリスト
    vertical_df_list = []

    # 指定された日数分ループ（1日目〜days日目まで）
    for day in range(1, days + 1):
        # 各日付ごとのフォルダパスを構築（例：input_base_folder/day1）
        input_folder = os.path.join(input_base_folder, f"day{day}")

        # フォルダが存在しない場合はスキップして次へ
        if not os.path.isdir(input_folder):
            print(f"スキップ: フォルダが見つかりません -> {input_folder}")
            continue

        # フォルダ内のCSVファイル一覧を取得
        csv_files = glob.glob(os.path.join(input_folder, "*.csv"))

        # 各CSVから抽出した列を格納するリスト
        df_list = []

        for file in csv_files:
            # CSVを読み込み（文字コードcp932 = Shift_JISに対応）
            df = pd.read_csv(file, encoding='cp932')

            # 列数が1列以下ならスキップ（2列以上必要）
            if df.shape[1] < 2:
                print(f"スキップ: 列数が足りない -> {file}")
                continue

            # 各CSVを最大665行に制限
            df = df.iloc[:665]

            # 2列目（index=1）のみを抽出
            second_col = df.iloc[:, [1]]

            # 抽出した列をリストに追加
            df_list.append(second_col)

        # 有効なCSVが1つもなかった場合、スキップ
        if not df_list:
            print(f"スキップ: 有効なCSVがありません -> {input_folder}")
            continue

        # 同日のCSVから抽出した2列目だけを横に連結（列方向）
        combined_df = pd.concat(df_list, axis=1)

        # 結果を日別リストに追加（後で縦に連結）
        vertical_df_list.append(combined_df)

    # 最後に全日分を縦に結合して最終データフレームに
    if vertical_df_list:
        final_df = pd.concat(vertical_df_list, axis=0, ignore_index=True)

        # 列名を「道路1」「道路2」…と自動で付ける
        final_headers = [f"道路{i+1}" for i in range(final_df.shape[1])]
        final_df.columns = final_headers

        # CSVとして保存（Shift_JIS形式、インデックスなし）
        final_df.to_csv(final_output_path, index=False, encoding='cp932')
        print(f"\n✅ 完了: {final_output_path}")
    else:
        # 有効なデータが一切なかった場合の警告
        print("⚠ 統合するデータがありませんでした。")


# ======================
# ファイル統合 (流入量)
# ======================
def combine_csv_ryuunyuu(input_base_folder_ryunyu, output_folder_ryunyu, days):
    # 出力フォルダを確実に作成しておく
    os.makedirs(output_folder_ryunyu, exist_ok=True)

    for day in range(1, days + 1):
        input_folder = os.path.join(input_base_folder_ryunyu, f"day{day}")
        output_filename = f"day{day}_車両流入.csv"
        output_path = os.path.join(output_folder_ryunyu, output_filename)

        # 対象フォルダが存在しない場合はスキップ
        if not os.path.isdir(input_folder):
            print(f"スキップ: フォルダが見つかりません -> {input_folder}")
            continue

        # フォルダ内のCSVファイルを取得
        csv_files = glob.glob(os.path.join(input_folder, "*.csv"))

        # データフレームのリスト
        df_list = []
        for file in csv_files:
            df = pd.read_csv(file)
            if df.shape[1] < 2:
                print(f"スキップ: 列数が足りない -> {file}")
                continue
            second_col = df.iloc[:, [1]]
            second_col.columns = [os.path.basename(file)]
            df_list.append(second_col)

        if not df_list:
            print(f"スキップ: 有効なCSVがありません -> {input_folder}")
            continue

        # 横方向に結合して保存
        combined_df = pd.concat(df_list, axis=1)
        combined_df.to_csv(output_path, index=False)
        print(f"✅ 完了: {output_path}")


# ======================
# ファイル移動 & リネーム
# ======================
def move_and_rename_rou_file_with_day(source_path, destination_folder, day):
    if not os.path.isfile(source_path):
        print(f"⚠ エラー: ファイルが存在しません → {source_path}")
        return

    # 常に .rou.xml を保持
    new_name = f"{day}日目.rou.xml"

    os.makedirs(destination_folder, exist_ok=True)
    destination_path = os.path.join(destination_folder, new_name)

    try:
        shutil.move(source_path, destination_path)
        print(f"✅ 移動完了: {destination_path}")
    except Exception as e:
        print(f"❌ エラー: {e}")


# ======================
# 実行
# ======================
if __name__ == "__main__":
    time_settings = [
        (1300, 4, 5), #準備   0    ~ 1300
        (3600, 4, 4), #1     1300  ~ 4900
        (3600, 4, 4), #2     4900  ~ 8500
        (3600, 4, 4), #3     8500  ~ 12100
        (3600, 3, 3), #4     12100 ~ 15700
        (3600, 3, 3), #5     15700 ~ 19300
        (3600, 2, 2), #6     19300 ~ 22900
        (3600, 2, 2), #7     22900 ~ 26500
        (3600, 1, 1), #8 大  26500 ~ 30100
        (3600, 1, 1), #9 大  30100 ~ 33700
        (3600, 1, 1), #10大  33700 ~ 37300
        (3600, 2, 2), #11    37300 ~ 40900
        (3600, 2, 2), #12    40900 ~ 44500
        (3600, 2, 2), #13    44500 ~ 48100 (切り替わり)
        (3600, 2, 2), #14    48100 ~ 51700
        (3600, 2, 2), #15    51700 ~ 55300
        (3600, 2, 2), #16    55300 ~ 58900
        (3600, 2, 2), #17    58900 ~ 62500
        (3600, 1, 1), #18大  62500 ~ 66100
        (3600, 1, 1), #19大  66100 ~ 69700
        (3600, 1, 1), #20大  69700 ~ 73300
        (3600, 2, 2), #21    73300 ~ 76900
        (3600, 2, 2), #22    76900 ~ 80500
        (3600, 2, 2), #23    80500 ~ 84100
        (3600, 3, 3), #24    84100 ~ 87700
    ]


# ======================
# 実行(テスト用)
# ======================
# if __name__ == "__main__":
#     time_settings = [
#         (1300, 4, 5),
#         (300, 5, 5),
#         (300, 5, 5),
#         (300, 4, 4),
#         (300, 3, 3),
#         (300, 2, 2),
#         (300, 1, 1),
#     ]

    warmup_time = time_settings[0][0]  # 準備時間を取得
    days = 1
    for day in range(1, days + 1):
        print(f"\n=== Day {day}: 車両生成開始 ===")
        generate_vehicle_routes(time_settings, day) #　車両生成
        print(f"=== Day {day}: シミュレーション開始 ===")
        run_simulation_with_signal_logging(day, warmup_time=warmup_time) #シミュレーション
        print(f"=== Day {day} まで統合開始 ===")
        combine_csv_vertically(machi_folder, output_filename, days) #待ち台数統合
        combine_csv_ryuunyuu(input_base_folder_ryunyu, output_folder_ryunyu, days) #流入量統合
        print(f"=== Day {day} を移動開始 ===")
        # move_and_rename_rou_file_with_day(source, destination, day) #rouファイル移動 & リネーム
        print(f"=== Day {day}: 完了 ===")
        
