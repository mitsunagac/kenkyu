import sys
import random
import csv
import sumolib
import traci
import os
import math
import xml.etree.ElementTree as ET
import pandas as pd
import glob
import re
import shutil
from collections import deque
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ======================
# 車両生成設定
# ======================
# netfile = r"C:\Users\Tsukasa\Desktop\研究\計測\toyama_shouwa_offset_smz.net.xml" #オフセット設定あり
# netfile = r"C:\Users\Tsukasa\Desktop\研究\計測\toyama_shouwa_no_offset.net.xml" #オフセット設定なし
netfile = r"C:\Users\Tsukasa\Desktop\研究\計測\shin_11.12shuusei.net.xml" #新環境(11/25)

to_edge_csv = r"C:\Users\Tsukasa\Desktop\研究\計測\output\交通環境調査\edges_with_2_or_more_lanes2.csv"
highway_edge_csv = to_edge_csv
rou_outputfile = r"C:\Users\Tsukasa\Desktop\研究\計測\random_trips_with_highway.rou.xml"
generation_log_base = r"C:\Users\Tsukasa\Desktop\研究\計測\output\発生台数\車両発生記録_day{}.csv"
kakuritsu_log_base = r"C:\Users\Tsukasa\Desktop\研究\計測\output\確率変化地点\確率変化地点_day{}.csv"

other_edge_csv = r"C:\Users\Tsukasa\Desktop\研究\計測\output\交通環境調査\edges_with_1_lane2.csv"

net = sumolib.net.readNet(Path(r"C:\Users\Tsukasa\Desktop\研究\計測\toyama_shouwa_offset_smz.net.xml").as_uri())

# ===== 朝・昼・夜 各時間帯の出発地エリア =====
# グループ定義（※from_edge 選定のために1回だけ定義すればOK）
group1_1 = ['start'] #南
group1_2 = ['-E82'] #南_東
group1_3 = ['E67'] #南_西

group2_1 = ['-E49']           #北
group2_2 = ['622541624#9']    #北_西
group2_3 = ['155381432#17']   #北_東

used_edges = set(group1_1 + group1_2 + group2_1 + group2_2)

# ======= 修正 ==================================
# 時間帯ごとの第1層の重み（南・北・その他）
# top_level_weights_by_time = [
#     [0.4, 0.3, 0.3],  # 深夜
#     [0.4, 0.3, 0.35],  # 早朝
#     [0.4, 0.3, 0.3],  # 朝 (ピーク)
#     [0.275, 0.225, 0.5],  # 昼
#     [0.275, 0.225, 0.5],  # 昼過ぎ
#     [0.26, 0.24, 0.5],  # 夕方 (ピーク)
#     [0.275, 0.225, 0.5],  # 夜
# ]
# # 第2層サブグループの重み
# group1_weights = [0.4, 0.3, 0.3]  # 南内の比率
# group2_weights = [0.4, 0.3, 0.3]  # 北内の比率
# ===============================================

# 時間帯ごとの第1層の重み（南・北・その他）
top_level_weights_by_time = [
    [0.4, 0.3, 0.3],  # 深夜
    [0.4, 0.3, 0.35],  # 早朝
    [0.4, 0.3, 0.3],  # 朝 (ピーク)
    [0.375, 0.325, 0.3],  # 昼
    [0.375, 0.325, 0.3],  # 昼過ぎ
    [0.36, 0.34, 0.3],  # 夕方 (ピーク)
    [0.375, 0.35, 0.3],  # 夜
]

# 第2層サブグループの重み
group1_weights = [0.5, 0.25, 0.25]  # 南内の比率
group2_weights = [0.6, 0.2, 0.2]  # 北内の比率

# ===== 朝・昼・夜 各時間帯の目的地地エリア =====
# 目的地設定
to_group1 = ['E49','-622541624#9','-155381437#0','976825362#0','E82', '-E67']
to_group2 = ['976825362#0', '-E67', 'E82','E49','-155381437#0', '-622541624#9']
to_group3 = ['976825362#0', 'E49']

# ====== 右左折率 =============
to_group1_weights_by_time = [
    [0.623, 0.07, 0.007, 0.1, 0.1, 0.1],  # 深夜
    [0.623, 0.07, 0.007, 0.1, 0.1, 0.1],  # 深夜
    [0.623, 0.07, 0.007, 0.1, 0.1, 0.1],  # 深夜
    [0.63, 0.07, 0, 0.1, 0.1, 0.1],
    [0.63, 0.07, 0, 0.1, 0.1, 0.1],
    [0.63, 0.07, 0, 0.1, 0.1, 0.1],
    [0.63, 0.07, 0, 0.1, 0.1, 0.1],
]

to_group2_weights_by_time = [
    [0.63, 0.07, 0, 0.1, 0.1, 0.1],
    [0.63, 0.07, 0, 0.1, 0.1, 0.1],
    [0.63, 0.07, 0, 0.1, 0.1, 0.1],
    [0.63, 0.07, 0, 0.1, 0.1, 0.1],
    [0.63, 0.07, 0, 0.1, 0.1, 0.1],
    [0.623, 0.07, 0.007, 0.1, 0.1, 0.1],  # 深夜
    [0.623, 0.07, 0.007, 0.1, 0.1, 0.1],  # 深夜
]


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
# 複数信号監視設定（エッジ単位で保存）
# ======================
TRAFFIC_LIGHT_CONFIG = {
    "E": [{"id": "2", "edges": ["E40", "-E22"], "red": "yyyrrrryyyrrrr", "green": "rrrrrrrrrrrrrr"},],
    "F": [{"id": "3", "edges":  ["E3", "E1", "-E34", "E35"], "red": "yyyrrrryyyrrrr", "green": "rrrrrrrrrrrrrr"},],
    "I": [{"id": "4", "edges": ["E57"], "red": "rrrryyyrrrryyy", "green": "rrrrrrrrrrrrrr"},],
    "J": [{ "id": "1", "edges": ["E24", "-E23", "-E33"], "red": "rrryrrrrrrryrrrr", "green": "rrrrrrrrrrrrrrrr"}, #南
          { "id": "5", "edges": ["E37", "E36"], "red": "rrrrrrryrrrrrrry","green": "rrrrrrrrrrrrrrrr"}, #西
          { "id": "6", "edges": ["-E25", "-E69"], "red": "rrrrrrryrrrrrrry","green": "rrrrrrrrrrrrrrrr"},], #東
    "K": [{"id": "7", "edges": ["155398822#14", "155398822#13"], "red": "rrrrrryy", "green": "rrrrrrrr"},],
    "M": [{"id": "8", "edges": ["E31", "E30"], "red": "yyyrrrryyyrrrr", "green": "rrrrrrrrrrrrrr"},],
    "N": [{"id": "9", "edges": ["E20", "E19"], "red": "yyyrrrryyyrrrr", "green": "rrrrrrrrrrrrrr"},],
}

TRAFFIC_LIGHT_CONFIG_2 = {
    "E": [{"id": "2", "edges": ["E40", "-E22"], "red": "rrryyyyrrryyyy", "green": "rrrrrrrrrrrrrr"},],
    "F": [{"id": "3", "edges":  ["E3", "E1", "-E34", "E35"], "red": "rrryyyyrrryyyy", "green": "rrrrrrrrrrrrrr"},],
    "I": [{"id": "4", "edges": ["E57"], "red": "yyyyrrryyyyrrr", "green": "rrrrrrrrrrrrrr"},],
    "J": [{ "id": "1", "edges": ["E24", "-E23", "-E33"], "red": "rrrryyyyrrrryyyy", "green": "rrrrrrrrrrrrrrrr"}, #南
          { "id": "5", "edges": ["E37", "E36"], "red": "yyyyrrrryyyyrrrr","green": "rrrrrrrrrrrrrrrr"}, #西
          { "id": "6", "edges": ["-E25", "-E69"], "red": "yyyyrrrryyyyrrrr","green": "rrrrrrrrrrrrrrrr"},], #東
    "K": [{"id": "7", "edges": ["155398822#14", "155398822#13"], "red": "yyyyyyrr", "green": "rrrrrrrr"},],
    "M": [{"id": "8", "edges": ["E31", "E30"], "red": "rrryyyyrrryyyy", "green": "rrrrrrrrrrrrrr"},],
    "N": [{"id": "9", "edges": ["E20", "E19"], "red": "rrryyyyrrryyyy", "green": "rrrrrrrrrrrrrr"},],
}



output_dir_base = r"C:\Users\Tsukasa\Desktop\研究\計測\output\待ち台数"
output_dir_base_2 = r"C:\Users\Tsukasa\Desktop\研究\計測\output\待ち台数_中間"
sumocfg_file = r"C:\Users\Tsukasa\Desktop\研究\計測\toyama_shouwa.sumocfg"
SUMO_CMD = ["sumo-gui", "-c", sumocfg_file, "--start", "--quit-on-end"]


# ======================
# ファイル統合設定
# ======================
merged_output_dir = r"C:\Users\Tsukasa\Desktop\研究\計測\output\待ち台数_遅れ時間"
merged_output_dir_1 = r"C:\Users\Tsukasa\Desktop\研究\計測\output\待ち台数_比較"

machi_folder = r"C:\Users\Tsukasa\Desktop\研究\計測\output\待ち台数"
output_filename = "all_days_待ち_統合.csv"

input_dir_base_3 = r"C:\Users\Tsukasa\Desktop\研究\計測\output\待ち台数_比較"  # CSVファイルが入っているフォルダに変更してください
output_dir_base_3 = r"C:\Users\Tsukasa\Desktop\研究\計測\output\遅れ時間"
merged_output_dir_3 = r"C:\Users\Tsukasa\Desktop\研究\計測\output\遅れ時間\統合"

# ======================
# ファイル移動設定
# ======================
source = r"C:\Users\Tsukasa\Desktop\研究\計測\random_trips_with_highway.rou.xml"
destination = r"C:\Users\Tsukasa\Desktop\研究\計測\output\生成された車"

# ======================
# 車両生成関数
# ======================
# ===== 幹線内だけで到達可能な経路を探索するBFS関数 =====
def find_path_on_highways(from_edge_id, to_edge_id, net, highway_edges):
    # 幹線道路エッジのみを通る経路をBFSで探索する
    visited = set()
    queue = deque([[from_edge_id]])  # 探索キューに初期状態を入れる

    while queue:
        path = queue.popleft()
        current_edge_id = path[-1]  # 現在のエッジ

        if current_edge_id == to_edge_id:
            return [net.getEdge(e) for e in path]  # 到達成功

        if current_edge_id in visited:
            continue
        visited.add(current_edge_id)

        try:
            current_edge = net.getEdge(current_edge_id)
            for succ_edge in current_edge.getOutgoing():
                succ_id = succ_edge.getID()
                if succ_id in highway_edges and succ_id not in visited:
                    queue.append(path + [succ_id])  # 有効な次のエッジをキューに追加
        except:
            continue

    return None  # 到達できない場合

# ===== 車両ルート生成のメイン関数 =====
def generate_vehicle_routes(all_time_settings, day, generation_range=None):
    # 各種出力ファイル準備
    generation_log_csv = generation_log_base.format(day)
    os.makedirs(os.path.dirname(generation_log_csv), exist_ok=True)
    kakuritsu_log_csv = kakuritsu_log_base.format(day)
    os.makedirs(os.path.dirname(kakuritsu_log_csv), exist_ok=True)

    # SUMOネットワーク読み込み
    net = sumolib.net.readNet(Path(netfile).as_uri())
    all_edges = [e.getID() for e in net.getEdges() if not e.getID().startswith(":")]

    # 目的地候補エッジ読み込み
    all_to_edges = []
    with open(to_edge_csv, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            edge_id = row['edge_id'].strip().strip("'\"")
            if edge_id:
                all_to_edges.append(edge_id)

    # 幹線エッジ読み込み
    highway_edges = []
    with open(highway_edge_csv, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            edge_id = row['edge_id'].strip().strip("'\"")
            if edge_id:
                highway_edges.append(edge_id)

    # 一般道（1車線）読み込み（group3で使用）
    otherway_edges = []
    with open(other_edge_csv, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            edge_id = row['edge_id'].strip().strip("'\"")
            if edge_id:
                otherway_edges.append(edge_id)

    # 幹線から到達可能な目的地を事前に探索して記録
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

    # # グループ3: group1・2 に属さない残りの幹線道路
    # to_group3 = [e for e in highway_edges if e not in to_group1 and e not in to_group2]

    # ===== 目的地選定関数 =====
    def pick_destination(from_edge, selected_top_group, is_from_highway, best_hwy, net, highway_edges, reachable_edges_from_highways, time_index):
        # グループに応じて候補と重みを設定
        if selected_top_group == 0:
            candidates = to_group1
            weights = to_group1_weights_by_time[time_index]
        elif selected_top_group == 1:
            candidates = to_group2
            weights = to_group2_weights_by_time[time_index]
        else:
            candidates = to_group3
            weights = None

        # 一般道出発：最寄り幹線（best_hwy）から行ける目的地を探す
        if not is_from_highway and best_hwy:
            reachable_targets = [
                e for e in candidates
                if e != from_edge and find_path_on_highways(best_hwy, e, net, set(highway_edges))
            ]
            if not reachable_targets:
                return None
            if weights:
                filtered_weights = [weights[candidates.index(e)] for e in reachable_targets if e in candidates]
                return random.choices(reachable_targets, weights=filtered_weights, k=1)[0]
            else:
                return random.choice(reachable_targets)

        # 幹線出発：到達可能なエッジから重みに基づき選択
        filtered = [e for e in candidates if e in reachable_edges_from_highways and e != from_edge]
        if not filtered:
            return None
        if weights:
            filtered_weights = [weights[candidates.index(e)] for e in filtered if e in candidates]
            return random.choices(filtered, weights=filtered_weights, k=1)[0]
        else:
            return random.choice(filtered)

    # ===== 車両生成処理 =====
    valid_routes = []
    vehicle_index = 0
    current_time = 0
    vehicle_counts = []

    # 測定用
    maji_count = 0
    maji_count_1 = 0
    maji_count_2 = 0
    maji_count_3 = 0
    maji_count_4 = 0
    maji_count_5 = 0
    all_maji_count = 0
    vehicle_counts = []

    generation_set = set(generation_range) if generation_range is not None else None

    for period_index, (sim_duration, depart_min, depart_max) in enumerate(all_time_settings):
        # 準備（index=0）以外は指定範囲のみ車両生成、それ以外は時刻だけ進める
        if generation_set is not None and period_index != 0 and period_index not in generation_set:
            vehicle_counts.append((period_index + 1, 0))
            continue

        time_limit = current_time + sim_duration
        count_this_period = 0
        time_index = get_time_index(period_index)  # 時間帯インデックス
        top_weights = top_level_weights_by_time[time_index]  # 南・北・その他の重み

        group3 = [e for e in otherway_edges]  # グループ3：その他の出発地
        from_area_choices = [group1_1 + group1_2, group2_1 + group2_2, group3]

        while current_time <= time_limit:
            # 出発グループ（トップ層）を選ぶ
            top_group_index = random.choices([0, 1, 2], weights=top_weights, k=1)[0]

            # サブグループ（第2層）を選ぶ
            if top_group_index == 0:
                sub_group = random.choices([group1_1, group1_2, group1_3], weights=group1_weights, k=1)[0]
            elif top_group_index == 1:
                sub_group = random.choices([group2_1, group2_2, group2_3], weights=group2_weights, k=1)[0]
            else:
                sub_group = group3

            from_edge = random.choice(sub_group)  # 出発地点エッジを選択
            is_from_highway = from_edge in highway_edges  # 幹線出発かどうか判定

            # ==== 幹線出発処理 ====
            if is_from_highway:
                to_edge = pick_destination(from_edge, top_group_index, True, None, net, highway_edges, reachable_edges_from_highways, time_index)
                if not to_edge or from_edge == to_edge:
                    print(f"[⚠️目的地エラー] from: {from_edge}, to: {to_edge} （目的地が無効か同一）")
                    continue
                path_on_highway = find_path_on_highways(from_edge, to_edge, net, set(highway_edges))
                if not path_on_highway:
                    print(f"[⚠️BFS失敗] 幹線: {from_edge} → to: {to_edge} への経路が見つかりません")
                    continue
                final_route = path_on_highway

            # ==== 一般道出発処理 ====
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
                    print(f"[⚠️接続失敗] from: {from_edge} → 幹線に接続できません")
                    continue

                to_edge = pick_destination(from_edge, top_group_index, False, best_hwy, net, highway_edges, reachable_edges_from_highways, time_index)
                if not to_edge or from_edge == to_edge:
                    print(f"[⚠️目的地エラー] from: {from_edge}, to: {to_edge} （目的地が無効か同一）")
                    continue

                path_to_dest = find_path_on_highways(best_hwy, to_edge, net, set(highway_edges))
                if not path_to_dest:
                    print(f"[⚠️BFS失敗] 幹線: {best_hwy} → to: {to_edge} への経路が見つかりません")
                    continue

                if path_to_highway[-1].getID() == path_to_dest[0].getID():
                    path_to_dest = path_to_dest[1:]

                final_route = list(path_to_highway) + path_to_dest
                print(f"[✅幹線経由] from: {from_edge} → 幹線: {best_hwy} → to: {to_edge}")

            # 経路登録
            route_ids = [e.getID() for e in final_route]
            valid_routes.append((vehicle_index, current_time, route_ids))
            print(f"⏱ 車両{vehicle_index} を {current_time} 秒目に生成中...")

            # 特定のエッジからの出発をカウント (朝)
            if from_edge == "start": 
                maji_count += 1 
                print("🎯 出発地に変えた場所が選ばれました")
            if from_edge == "E67": 
                maji_count_1 += 1
                print("🎯 出発地に変えた場所が選ばれました")
            if from_edge == "-E82": 
                maji_count_2 += 1
                print("🎯 出発地に変えた場所が選ばれました")

            # 特定のエッジからの出発をカウント (夜)
            if from_edge == "-E49": 
                maji_count_3 += 1
                print("🔹 出発地に変えた場所が選ばれました")
            if from_edge == "622541624#9": 
                maji_count_4 += 1
                print("🔹 出発地に変えた場所が選ばれました")
            if from_edge == "155381432#17": 
                maji_count_5 += 1
                print("🔹 出発地に変えた場所が選ばれました")

            # 時刻・カウント更新
            vehicle_index += 1
            current_time += random.randint(depart_min, depart_max)
            count_this_period += 1

        # この時間帯で生成した車両数を保存
        vehicle_counts.append((period_index + 1, count_this_period))


    # rou.xml出力
    with open(rou_outputfile, "w") as f:
        f.write("<routes>\n")
        f.write('  <vType id="car" accel="2.0" decel="4.5" maxSpeed="16.67" length="5" sigma="0.3"/>\n')
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
    print(f"🎯 南から{maji_count} 台, 東から{maji_count_2} 台, 西から{maji_count_1} 台")
    print(f"🎯 北から{maji_count_3} 台, 北西から{maji_count_4} 台, 北東から{maji_count_5} 台")
    print(f"✅ 確率変更点から{all_maji_count} 台, 生成されました")




# ======================
# 信号機ごとの各方向エッジごとに計測 & 流入量測定
# ======================
def run_simulation_with_signal_logging(day, warmup_time=0):
    # ==== 出力ディレクトリの準備（それぞれ別） ====
    output_dir_1 = os.path.join(output_dir_base, f"day{day}")
    output_dir_2 = os.path.join(output_dir_base_2, f"day{day}")
    os.makedirs(output_dir_1, exist_ok=True)
    os.makedirs(output_dir_2, exist_ok=True)

    csv_files_1 = {}
    writers_1 = {}
    csv_files_2 = {}
    writers_2 = {}

    # ==== TRAFFIC_LIGHT_CONFIGごとにCSVファイルを準備 ====
    for tl_id, configs in TRAFFIC_LIGHT_CONFIG.items():
        for config in configs:
            filename = f"道路{config['id']}_day{day}.csv"
            path = os.path.join(output_dir_1, filename)
            f = open(path, mode='w', newline='')
            w = csv.writer(f)
            w.writerow(["Time (s)", "Waiting Cars"])
            csv_files_1[config['id']] = f
            writers_1[config['id']] = w

    # ==== TRAFFIC_LIGHT_CONFIG_2ごとにCSVファイルを準備 ====
    for tl_id, configs in TRAFFIC_LIGHT_CONFIG_2.items():
        for config in configs:
            filename = f"道路{config['id']}_day{day}.csv"
            path = os.path.join(output_dir_2, filename)
            f = open(path, mode='w', newline='')
            w = csv.writer(f)
            w.writerow(["Time (s)", "Waiting Cars"])
            csv_files_2[config['id']] = f
            writers_2[config['id']] = w

    # ==== SUMOシミュレーション開始 ====
    traci.start(SUMO_CMD)

    # ==== 状態管理用 ====
    prev_states_1 = {tl_id: ["" for _ in TRAFFIC_LIGHT_CONFIG[tl_id]] for tl_id in TRAFFIC_LIGHT_CONFIG}
    prev_states_2 = {tl_id: ["" for _ in TRAFFIC_LIGHT_CONFIG_2[tl_id]] for tl_id in TRAFFIC_LIGHT_CONFIG_2}

    pending_counts_1 = {}
    pending_counts_2 = {}

    # ==== メインループ ====
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        sim_time = traci.simulation.getTime()

        # ==== CONFIG 1（道路別タイミング）====
        for tl_id, configs in TRAFFIC_LIGHT_CONFIG.items():
            for i, config in enumerate(configs):
                try:
                    current = traci.trafficlight.getRedYellowGreenState(tl_id)
                    prev = prev_states_1[tl_id][i]
                    key = (tl_id, i)

                    if sim_time >= warmup_time and prev == config['red'] and current == config['green']:
                        pending_counts_1[key] = sim_time

                    if key in pending_counts_1 and sim_time >= pending_counts_1[key]:
                        total_count = 0
                        for edge in config['edges']:
                            for veh_id in traci.edge.getLastStepVehicleIDs(edge):
                                if traci.vehicle.getSpeed(veh_id) <= 0:
                                    total_count += 1

                        try:
                            id_int = int(config['id'])
                            if id_int in [1, 2, 3, 8, 9]:
                                total_count = math.ceil(total_count / 2)
                            elif id_int in [4, 5, 6, 7]:
                                total_count = math.ceil(total_count / 2)
                        except:
                            pass

                        writers_1[config['id']].writerow([sim_time, total_count])
                        del pending_counts_1[key]

                    prev_states_1[tl_id][i] = current

                except Exception as e:
                    print(f"⚠ Error at {tl_id} ({config['id']}) - CONFIG 1: {e}")

        # ==== CONFIG 2（道路別タイミング）====
        for tl_id, configs in TRAFFIC_LIGHT_CONFIG_2.items():
            for i, config in enumerate(configs):
                try:
                    current = traci.trafficlight.getRedYellowGreenState(tl_id)
                    prev = prev_states_2[tl_id][i]
                    key = (tl_id, i)

                    if sim_time >= warmup_time and prev == config['red'] and current == config['green']:
                        try:
                            id_int = int(config['id'])
                            if id_int in [1]:
                                delay = 28
                            elif id_int in [5, 6]:
                                delay = 45
                            else:
                                delay = 21

                        except Exception as e:
                            print(f"⚠ エラー発生: {e}")

                        pending_counts_2[key] = sim_time + delay

                    if key in pending_counts_2 and sim_time >= pending_counts_2[key]:
                        total_count = 0
                        for edge in config['edges']:
                            for veh_id in traci.edge.getLastStepVehicleIDs(edge):
                                if traci.vehicle.getSpeed(veh_id) <= 0:
                                    total_count += 1

                        try:
                            id_int = int(config['id'])
                            if id_int in [1, 2, 3, 8, 9]:
                                total_count = math.ceil(total_count / 2)
                            elif id_int in [4, 5, 6, 7]:
                                total_count = math.ceil(total_count / 2)
                        except:
                            pass

                        writers_2[config['id']].writerow([sim_time, total_count])
                        del pending_counts_2[key]

                    prev_states_2[tl_id][i] = current

                except Exception as e:
                    print(f"⚠ Error at {tl_id} ({config['id']}) - CONFIG 2: {e}")

    # ==== 終了処理 ====
    for f in csv_files_1.values():
        f.close()
    for f in csv_files_2.values():
        f.close()

    traci.close()
    print(f"✅ Day {day}: 待機車両記録を完了しました → {output_dir_1} / {output_dir_2}")

#====== ファイル統合 (遅れ時間) ====================================================

# ======================
# ファイル統合(待ち台数_遅れ時間考慮)
# ======================
def merge_waiting_csvs(day):
    # 各ディレクトリ
    dir1 = os.path.join(output_dir_base, f"day{day}")
    dir2 = os.path.join(output_dir_base_2, f"day{day}")
    merged_dir = os.path.join(merged_output_dir, f"day{day}")
    os.makedirs(merged_dir, exist_ok=True)

    # ファイル名一覧（output_dir_base側を基準に）
    filenames = [f for f in os.listdir(dir1) if f.endswith(".csv")]

    for filename in filenames:
        file1 = os.path.join(dir1, filename)
        file2 = os.path.join(dir2, filename)

        if not (os.path.exists(file1) and os.path.exists(file2)):
            print(f"⚠ {filename} が両方存在しません。スキップします。")
            continue

        # データ読み込み＆インデックスリセット
        df1 = pd.read_csv(file1).reset_index(drop=True)
        df2 = pd.read_csv(file2).reset_index(drop=True)

        # --- 最初の1行目だけ比較して処理する ---
        if len(df1) > 0 and len(df2) > 0:
            if df1.iloc[0, 0] < df2.iloc[0, 0]:
                df1 = df1.iloc[1:].reset_index(drop=True)  # df1の最初の行をスキップ

        # 行数を合わせる
        max_len = min(len(df1), len(df2))

        # 交互に結合：df1 - df2 → df1 → ...
        merged_rows = []
        for i in range(max_len):
            diff_row = df1.iloc[i] - df2.iloc[i]
            merged_rows.append(df2.iloc[i])
            merged_rows.append(diff_row)


        # DataFrame化して保存
        merged_df = pd.DataFrame(merged_rows)
        output_path = os.path.join(merged_dir, filename)
        merged_df.to_csv(output_path, index=False)
        print(f"✅ {filename} を統合して保存しました → {output_path}")

    print(f"🎯 Day{day} の統合完了！")


# ======================
# ファイル統合(遅れ時間)
# ======================
def combine_csv_okurejikan(merged_output_dir, output_filename, days):
    # 出力先のフォルダ「統合」を作成（存在しない場合のみ）
    output_folder = os.path.join(merged_output_dir, "統合")
    os.makedirs(output_folder, exist_ok=True)

    # 結合後のCSVファイルの保存パスを作成
    final_output_path = os.path.join(output_folder, output_filename)

    # 全日分のデータフレームを格納するリスト
    vertical_df_list = []

    # 指定された日数分ループ（1日目〜days日目まで）
    for day in range(1, days + 1):
        # 各日付ごとのフォルダパスを構築（例：input_base_folder/day1）
        input_folder = os.path.join(merged_output_dir, f"day{day}")

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
            df = df.iloc[:1330]

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

# ======================================================================================================

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
# 横統合(テスト)
# ======================
def merge_waiting_csvs_side_by_side(day):
    # 各ディレクトリ
    dir1 = os.path.join(output_dir_base, f"day{day}")
    dir2 = os.path.join(output_dir_base_2, f"day{day}")
    merged_dir = os.path.join(merged_output_dir_1, f"day{day}")
    os.makedirs(merged_dir, exist_ok=True)

    filenames = [f for f in os.listdir(dir1) if f.endswith(".csv")]

    for filename in filenames:
        file1 = os.path.join(dir1, filename)
        file2 = os.path.join(dir2, filename)

        if not (os.path.exists(file1) and os.path.exists(file2)):
            print(f"⚠ {filename} が両方存在しません。スキップします。")
            continue

        # ファイル読み込み（2列目のみ）
        df1 = pd.read_csv(file1, usecols=[1])
        df2 = pd.read_csv(file2, usecols=[1])

        # インデックスをリセット
        df1 = df1.reset_index(drop=True)
        df2 = df2.reset_index(drop=True)

        # カラム名を設定
        df2.columns = ["WaitingCars_39.5sec"]
        df1.columns = ["WaitingCars_2sec"]

        # 中間値の列を計算（df1 - df2）
        df_middle = pd.DataFrame()
        df_middle["後半"] = df1["WaitingCars_2sec"] - df2["WaitingCars_39.5sec"]

        # 横に結合
        merged_df = pd.concat([df2, df_middle], axis=1)

        # 出力
        output_path = os.path.join(merged_dir, filename)
        merged_df.to_csv(output_path, index=False)
        print(f"✅ {filename} を統合して保存しました（中間値付き）→ {output_path}")

    print(f"🎯 Day{day} の統合完了（中間値含む）！")

# ======================
# 待ち台数比較 (本命)
# ======================
def merge_all_days(merged_output_dir_3, output_filename="遅れ時間_統合_all_days.csv", add_day_column=True):
    """
    merged_output_dir_3/日ごと/ 内の遅れ時間_統合_dayX.csv を結合し、
    merged_output_dir_3/遅れ時間_統合_all_days.csv に出力。
    """
    all_dfs = []
    day_folder = os.path.join(merged_output_dir_3, "日ごと")

    if not os.path.exists(day_folder):
        print(f"⚠ 統合フォルダが見つかりません: {day_folder}")
        return

    for filename in sorted(os.listdir(day_folder)):
        match = re.match(r"遅れ時間_統合_day(\d+)\.csv", filename)
        if not match:
            continue

        filepath = os.path.join(day_folder, filename)

        try:
            df = pd.read_csv(filepath, encoding="utf-8-sig")
            all_dfs.append(df)
            print(f"✅ 読み込み成功: {filepath}")
        except Exception as e:
            print(f"❌ 読み込み失敗: {filepath} - {e}")

    if all_dfs:
        merged_all = pd.concat(all_dfs, ignore_index=True)
        output_path = os.path.join(merged_output_dir_3, output_filename)
        merged_all.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"✅ 結合出力完了: {output_path}")
    else:
        print("⚠ 結合対象が見つかりませんでした。")


def compare_and_merge_results(day, input_dir_base_3, output_dir_base_3, merged_output_dir_3):
    """
    1. 道路◯_day◯.csv を比較し Result を出力（遅れ時間_道路◯_◯day.csv）
    2. 各 day の結果を merged_output_dir_3/日ごと/ に保存
    3. 遅れ時間_統合_all_days.csv を再作成
    """
    input_dir = os.path.join(input_dir_base_3, f"day{day}")
    output_dir = os.path.join(output_dir_base_3, f"day{day}")
    daily_merge_dir = os.path.join(merged_output_dir_3, "日ごと")  # ✅ 共通フォルダ「日ごと」
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(daily_merge_dir, exist_ok=True)
    os.makedirs(merged_output_dir_3, exist_ok=True)

    result_columns = {}
    max_rows = 665

    if not os.path.exists(input_dir):
        print(f"❌ 入力フォルダが存在しません: {input_dir}")
        return

    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith(".csv"):
            print(f"📄 処理中ファイル: {filename}")
            match = re.match(r"道路(\d+)_day0*(\d+)\.csv", filename)
            if not match:
                print(f"⚠ 無視されました（形式外）: {filename}")
                continue

            road_number = int(match.group(1))
            day_number = match.group(2)

            input_path = os.path.join(input_dir, filename)
            output_filename = f"遅れ時間_道路{road_number}_{day_number}day.csv"
            output_path = os.path.join(output_dir, output_filename)

            try:
                df = pd.read_csv(input_path, header=None, skiprows=1, usecols=[0, 1])
                comparison_result = (
                    (df[0] < df[1]).astype(int) * 0 +
                    (df[0] == df[1]).astype(int) * 1 +
                    (df[0] > df[1]).astype(int) * 2
                )

                trimmed_result = comparison_result[:max_rows].reset_index(drop=True)

                output_df = pd.DataFrame({
                    "Left": df[0],
                    "Right": df[1],
                    "Result": comparison_result
                })
                output_df.to_csv(output_path, index=False, encoding='utf-8-sig')
                print(f"✅ 比較出力: {output_filename}")

                result_columns[f"道路{road_number}"] = trimmed_result

            except Exception as e:
                print(f"❌ エラー: {filename} - {e}")

    if result_columns:
        merged_df = pd.DataFrame(result_columns)
        merged_df = merged_df.reindex(range(max_rows))
        merged_day_path = os.path.join(daily_merge_dir, f"遅れ時間_統合_day{day}.csv")  # ✅ 「日ごと」フォルダへ
        merged_df.to_csv(merged_day_path, index=False, encoding='utf-8-sig')
        print(f"✅ 統合CSV出力（665行）: {merged_day_path}")

        # ✅ all_days ファイルも更新
        merge_all_days(merged_output_dir_3)
    else:
        print(f"⚠ 統合対象が1つも見つかりませんでした for day{day}")

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
        (1200, 8, 12), #準備   0    ~ 1300
        (3600, 8, 12), #1     1300  ~ 4900
        (3600, 8, 12), #2     4900  ~ 8500
        (3600, 8, 12), #3     8500  ~ 12100
        (3600, 8, 12), #4     12100 ~ 15700
        (3600, 8, 12), #5     15700 ~ 19300
        (3600, 2, 6), #6     19300 ~ 22900
        (3600, 1, 1), #7     22900 ~ 26500
        (3600, 1, 2), #8 大  26500 ~ 30100
        (3600, 1, 2), #9 大  30100 ~ 33700
        (3600, 1, 2), #10    33700 ~ 37300
        (3600, 1, 2), #11    37300 ~ 40900
        (3600, 1, 2), #12    40900 ~ 44500
        (3600, 1, 2), #13    44500 ~ 48100 (切り替わり)
        (3600, 1, 2), #14    48100 ~ 51700
        (3600, 1, 2), #15    51700 ~ 55300
        (3600, 1, 2), #16    55300 ~ 58900
        (3600, 1, 2), #17    58900 ~ 62500
        (3600, 1, 1), #18大  62500 ~ 66100
        (3600, 1, 2), #19大  66100 ~ 69700
        (3600, 1, 3), #20    69700 ~ 73300
        (3600, 1, 4), #21    73300 ~ 76900
        (3600, 1, 5), #22    76900 ~ 80500
        (3600, 2, 6), #23    80500 ~ 84100
        (3600, 4, 7), #24    84100 ~ 87700
    ]

# if __name__ == "__main__":
#     time_settings = [
#         (1200, 8, 12), #準備   0    ~ 1300
#         (3600, 8, 12), #1     1300  ~ 4900
#         (3600, 8, 12), #2     4900  ~ 8500
#         (3600, 8, 12), #3     8500  ~ 12100
#         (3600, 8, 12), #4     12100 ~ 15700
#         (3600, 8, 12), #5     15700 ~ 19300
#         (3600, 2, 6), #6     19300 ~ 22900
#         (3600, 2, 2), #7     22900 ~ 26500
#         (3600, 2, 2), #8 大  26500 ~ 30100
#         (3600, 2, 3), #9 大  30100 ~ 33700
#         (3600, 2, 3), #10    33700 ~ 37300
#         (3600, 2, 3), #11    37300 ~ 40900
#         (3600, 2, 3), #12    40900 ~ 44500
#         (3600, 2, 3), #13    44500 ~ 48100 (切り替わり)
#         (3600, 2, 3), #14    48100 ~ 51700
#         (3600, 2, 3), #15    51700 ~ 55300
#         (3600, 2, 3), #16    55300 ~ 58900
#         (3600, 2, 3), #17    58900 ~ 62500
#         (3600, 2, 2), #18大  62500 ~ 66100
#         (3600, 2, 3), #19大  66100 ~ 69700
#         (3600, 2, 4), #20    69700 ~ 73300
#         (3600, 2, 5), #21    73300 ~ 76900
#         (3600, 2, 6), #22    76900 ~ 80500
#         (3600, 3, 7), #23    80500 ~ 84100
#         (3600, 5, 8), #24    84100 ~ 87700
#     ]

    warmup_time = time_settings[0][0]  # 準備時間を取得

    # ======================================================
    # 車両を生成する時間帯の範囲（time_settingsのインデックス）
    # 準備（index=0）は常に生成。それ以外の範囲をここで指定。
    # 例: #10〜#22 → range(10, 23)
    #     #1〜#5  → range(1, 6)
    # ======================================================
    VEHICLE_GENERATION_RANGE = range(6, 19)  # #10 〜 #22

    days = 3
    for day in range(1, days + 1):
        print(f"\n=== Day {day}: 車両生成開始 ===")
        generate_vehicle_routes(time_settings, day, VEHICLE_GENERATION_RANGE) #　車両生成

        # print(f"=== Day {day}: シミュレーション開始 ===")
        # run_simulation_with_signal_logging(day, warmup_time=warmup_time) #シミュレーション

        # print(f"=== Day {day} まで統合開始 ===")
        # print (f"=== Day {day} : 遅れ時間_統合 ===")
        # merge_waiting_csvs(day) # 待ち台数＆遅れ時間 統合(日毎)
        # combine_csv_okurejikan(merged_output_dir, output_filename, days) # 待ち＆遅れ時間 統合(全日)

        # print(f"=== Day {day} : 待ち台数_統合 ===")
        # combine_csv_vertically(machi_folder, output_filename, days) # 総待ち台数 統合

        # print(f"=== Day {day}：比較処理開始 ===")
        # merge_waiting_csvs_side_by_side(day) #横統合 (遅れ時間_比較)
        # compare_and_merge_results(day, input_dir_base_3, output_dir_base_3, merged_output_dir_3)

        print(f"=== Day {day} を移動開始 ===")
        move_and_rename_rou_file_with_day(source, destination, day) #rouファイル移動 & リネーム
        
        print(f"=== Day {day}: 完了 ===")
        
