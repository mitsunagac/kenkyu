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
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ======================
# 車両生成設定
# ======================
# netfile = r"C:\Users\Tsukasa\Desktop\研究\計測\toyama_shouwa_offset_smz.net.xml" #オフセット設定あり
# netfile = r"C:\Users\Tsukasa\Desktop\研究\計測\toyama_shouwa_no_offset.net.xml" #オフセット設定なし
netfile = r"C:\Users\Tsukasa\Desktop\研究\計測\shin_11.12shuusei.net.xml" #新環境(11/25)

project_root = r"C:\Users\Tsukasa\Desktop\研究\計測"
vehicle_generation_data_root = os.path.join(project_root, "車両生成データ")
traffic_environment_dir = os.path.join(vehicle_generation_data_root, "交通環境調査")
generation_log_dir = os.path.join(vehicle_generation_data_root, "発生台数")
kakuritsu_log_dir = os.path.join(vehicle_generation_data_root, "確率変化地点")
generated_vehicle_dir = os.path.join(vehicle_generation_data_root, "★車両データ", "6月28日作成 (24hデータ)", "生成された車_北南20%削減_全日_2026-07-14")

to_edge_csv = os.path.join(traffic_environment_dir, "edges_with_2_or_more_lanes2.csv")
highway_edge_csv = to_edge_csv
rou_outputfile = os.path.join(project_root, "random_trips_with_highway.rou.xml")
generation_log_base = os.path.join(generation_log_dir, "車両発生記録_day{}.csv")
kakuritsu_log_base = os.path.join(kakuritsu_log_dir, "確率変化地点_day{}.csv")

other_edge_csv = os.path.join(traffic_environment_dir, "edges_with_1_lane2.csv")

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


# ======================
# サイクル長 時間帯別設定
# ======================
BASE_CYCLE_LENGTH = 120          # 基準サイクル長(s) ※.net.xml 上の現行値
PREP_CYCLE_LENGTH = 100          # 準備時間帯のサイクル長(s)
GREEN_PHASE_MIN_DURATION = 10    # この秒数以上のフェーズを「調整対象の青」とみなす（黄3s・右折矢印5s・全赤3sは対象外）

# 時間帯(時)ごとの目標サイクル長(s)
CYCLE_SCHEDULE_BY_HOUR = [
    (0, 7, 100),    # 0〜7時
    (7, 8, 130),    # 7〜8時 (朝ピーク)
    (8, 17, 110),   # 8〜17時
    (17, 19, 130),  # 17〜19時 (夕方ピーク)
    (19, 24, 100),  # 19〜24時
]


def get_target_cycle(sim_time, warmup_time):
    """シミュレーション時刻(s)から目標サイクル長(s)を返す。"""
    if sim_time < warmup_time:
        return PREP_CYCLE_LENGTH  # 準備時間帯
    hour = (sim_time - warmup_time) / 3600.0  # 準備終了を0時とした経過時間(時)
    for start_hour, end_hour, cycle in CYCLE_SCHEDULE_BY_HOUR:
        if start_hour <= hour < end_hour:
            return cycle
    return CYCLE_SCHEDULE_BY_HOUR[-1][2]  # 24時以降は最後の設定を継続


def build_scaled_logic(base_logic, target_cycle):
    """基準プログラムのフェーズ継続時間だけを組み替え、目標サイクル長のLogicを生成する。
    黄・全赤・右折矢印(短いフェーズ)は固定し、長い青フェーズに 120s との差分を均等配分する。
    フェーズ数・状態文字列・順序・番号は一切変えないため、計測ロジックには影響しない。"""
    phases = base_logic.phases
    base_cycle = sum(ph.duration for ph in phases)
    green_idxs = [i for i, ph in enumerate(phases) if ph.duration >= GREEN_PHASE_MIN_DURATION]
    if not green_idxs:
        return None  # 調整対象の青フェーズが無い
    per_green_delta = (target_cycle - base_cycle) / len(green_idxs)  # 均等配分（各 ΔC/2）

    new_phases = []
    for i, ph in enumerate(phases):
        next_phases = getattr(ph, "next", ())
        name = getattr(ph, "name", "")
        if i in green_idxs:
            new_duration = ph.duration + per_green_delta
            new_phases.append(traci.trafficlight.Phase(
                new_duration, ph.state, new_duration, new_duration, next_phases, name
            ))
        else:
            new_phases.append(traci.trafficlight.Phase(
                ph.duration, ph.state, ph.minDur, ph.maxDur, next_phases, name
            ))
    return traci.trafficlight.Logic(
        base_logic.programID,
        base_logic.type,
        base_logic.currentPhaseIndex,
        new_phases,
        base_logic.subParameter,
    )


organized_output_root = r"C:\Users\Tsukasa\Desktop\研究\計測\output\計測結果_北南20%削減_全日"
sumocfg_file = r"C:\Users\Tsukasa\Desktop\研究\計測\toyama_shouwa.sumocfg"
SUMO_CMD = ["sumo", "-c", sumocfg_file, "--start", "--quit-on-end"]

# 赤時間3分割計測の出力先（道路ごと・日ごとにサイクル別CSVを保存）
red_split_output_dir = os.path.join(organized_output_root, "赤時間3分割", "サイクル別")
# 全道路・全日を1つにまとめた統合CSVの出力先
red_split_merged_dir = os.path.join(organized_output_root, "赤時間3分割", "統合")


# ======================
# 赤時間3分割 計測設定
# ======================
# 各道路が「青」「黄」になるフェーズ番号（ネット接続から検証済み）。
# 赤サイクル = yellow_phase 終了 〜 次の green_phase 開始。
ROAD_PHASE_TARGETS = {
    "1": {"green_phase": 0, "yellow_phase": 1},  # J 主(南北)
    "2": {"green_phase": 0, "yellow_phase": 1},  # E 主
    "3": {"green_phase": 0, "yellow_phase": 1},  # F 主
    "4": {"green_phase": 0, "yellow_phase": 1},  # I 主
    "5": {"green_phase": 5, "yellow_phase": 6},  # J 従(西)
    "6": {"green_phase": 5, "yellow_phase": 6},  # J 従(東)
    "7": {"green_phase": 0, "yellow_phase": 1},  # K 主
    "8": {"green_phase": 0, "yellow_phase": 1},  # M 主
    "9": {"green_phase": 0, "yellow_phase": 1},  # N 主
}

# 赤時間の分割数
RED_SPLIT_COUNT = 3

# 道路ごとの停車台数の除数（既存踏襲で全道路 ÷2。1 にすれば実台数）
ROAD_COUNT_DIVISOR = {str(i): 2 for i in range(1, 10)}


# ======================
# ファイル移動設定
# ======================
source = rou_outputfile
destination = generated_vehicle_dir

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
                    continue
                path_on_highway = find_path_on_highways(from_edge, to_edge, net, set(highway_edges))
                if not path_on_highway:
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
                    continue

                to_edge = pick_destination(from_edge, top_group_index, False, best_hwy, net, highway_edges, reachable_edges_from_highways, time_index)
                if not to_edge or from_edge == to_edge:
                    continue

                path_to_dest = find_path_on_highways(best_hwy, to_edge, net, set(highway_edges))
                if not path_to_dest:
                    continue

                if path_to_highway[-1].getID() == path_to_dest[0].getID():
                    path_to_dest = path_to_dest[1:]

                final_route = list(path_to_highway) + path_to_dest

            # 経路登録
            route_ids = [e.getID() for e in final_route]
            valid_routes.append((vehicle_index, current_time, route_ids))

            # 特定のエッジからの出発をカウント (朝)
            if from_edge == "start":
                maji_count += 1
            if from_edge == "E67":
                maji_count_1 += 1
            if from_edge == "-E82":
                maji_count_2 += 1

            # 特定のエッジからの出発をカウント (夜)
            if from_edge == "-E49":
                maji_count_3 += 1
            if from_edge == "622541624#9":
                maji_count_4 += 1
            if from_edge == "155381432#17":
                maji_count_5 += 1

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

    all_maji_count = maji_count + maji_count_1 + maji_count_2 + maji_count_3 + maji_count_4 + maji_count_5




# ======================
# 信号機ごとの各方向エッジごとに計測 & 流入量測定
# ======================
def run_simulation_with_signal_logging(day, warmup_time=0):
    def open_csv_with_fallback(path, encoding=None):
        open_kwargs = {"mode": "w", "newline": ""}
        if encoding is not None:
            open_kwargs["encoding"] = encoding

        try:
            return open(path, **open_kwargs)
        except PermissionError:
            root, ext = os.path.splitext(path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            for number in range(1, 100):
                suffix = timestamp if number == 1 else f"{timestamp}_{number}"
                fallback_path = f"{root}_{suffix}{ext}"
                try:
                    f = open(fallback_path, **open_kwargs)
                    print(f"⚠ {path} が開けないため、別名で出力します → {fallback_path}")
                    return f
                except PermissionError:
                    continue
            raise

    # ==== 出力ディレクトリの準備 ====
    red_split_dir_day = os.path.join(red_split_output_dir, f"day{day}")
    os.makedirs(red_split_dir_day, exist_ok=True)

    def get_road_target(config):
        return ROAD_PHASE_TARGETS.get(str(config['id']))

    # ==== 対象道路(全9道路)ごとに3分割CSVを準備 ====
    red_split_files = {}    # (tl_id, i) -> file
    red_split_writers = {}  # (tl_id, i) -> writer
    for tl_id, configs in TRAFFIC_LIGHT_CONFIG.items():
        for i, config in enumerate(configs):
            if get_road_target(config) is None:
                continue
            filename = f"道路{config['id']}_day{day}.csv"
            path = os.path.join(red_split_dir_day, filename)
            f = open_csv_with_fallback(path, encoding='utf-8-sig')
            w = csv.writer(f)
            w.writerow([
                "サイクル数",
                "赤開始時刻(s)",
                "赤時間(s)",
                "区間1台数",
                "区間2台数",
                "区間3台数",
                "合計台数",
            ])
            red_split_files[(tl_id, i)] = f
            red_split_writers[(tl_id, i)] = w

    # ==== SUMOシミュレーション開始 ====
    traci.start(SUMO_CMD)

    # ==== サイクル長 時間帯別制御のセットアップ ====
    # ネット内の全120s信号を対象に、各目標サイクル長のスケール済みプログラムを事前生成する
    cycle_scaled_logics = {}  # tl_id -> {cycle_length: Logic}
    target_cycle_lengths = {PREP_CYCLE_LENGTH} | {c for _, _, c in CYCLE_SCHEDULE_BY_HOUR}
    for tl_id in traci.trafficlight.getIDList():
        try:
            base_logic = traci.trafficlight.getAllProgramLogics(tl_id)[0]
        except Exception as e:
            print(f"⚠ サイクル制御: {tl_id} のプログラム取得に失敗しました: {e}")
            continue
        base_cycle = sum(ph.duration for ph in base_logic.phases)
        if abs(base_cycle - BASE_CYCLE_LENGTH) > 0.5:
            continue  # 120s以外(90s信号など)は対象外
        scaled = {}
        for cycle_length in target_cycle_lengths:
            logic = build_scaled_logic(base_logic, cycle_length)
            if logic is not None:
                scaled[cycle_length] = logic
        if scaled:
            cycle_scaled_logics[tl_id] = scaled
    print(f"✅ サイクル長 時間帯別制御: 対象信号 {len(cycle_scaled_logics)} 個 / 目標長 {sorted(target_cycle_lengths)}s")
    current_applied_cycle = None  # まだ適用していない

    # ==== 状態管理用 ====
    prev_phases_1 = {tl_id: None for tl_id in TRAFFIC_LIGHT_CONFIG}
    STOP_SPEED_THRESHOLD = 0
    red_cycle_counts = {key: 0 for key in red_split_writers}
    active_red_cycles = {}

    def write_red_split(key, cycle_info):
        # 赤時間を3等分し、各区間で「初めて停車した」車両台数を数えて書き出す
        road_id = cycle_info["road_id"]
        red_duration = round(cycle_info["red_end_time"] - cycle_info["red_start_time"], 3)
        counts = [0] * RED_SPLIT_COUNT
        if red_duration > 0:
            third = red_duration / RED_SPLIT_COUNT
            for stop_time in cycle_info["stop_times"]:
                idx = int(stop_time // third)
                if idx >= RED_SPLIT_COUNT:
                    idx = RED_SPLIT_COUNT - 1  # 境界(=赤終了ちょうど)は最終区間へ
                counts[idx] += 1
        # 道路ごとの除数を適用（既存踏襲）
        divisor = ROAD_COUNT_DIVISOR.get(road_id, 1)
        counts = [math.ceil(c / divisor) for c in counts]
        red_split_writers[key].writerow(
            [cycle_info["cycle"], round(cycle_info["red_start_time"], 3), red_duration]
            + counts
            + [sum(counts)]
        )

    def record_stops(key, config, sim_time):
        # 赤サイクル中、対象エッジで初めて停車した車両の「赤開始からの経過時間」を記録
        cycle_info = active_red_cycles.get(key)
        if cycle_info is None:
            return

        stop_time_after_red = round(sim_time - cycle_info["red_start_time"], 3)
        for edge in config['edges']:
            for veh_id in traci.edge.getLastStepVehicleIDs(edge):
                if veh_id in cycle_info["recorded_vehicle_ids"]:
                    continue
                if traci.vehicle.getSpeed(veh_id) <= STOP_SPEED_THRESHOLD:
                    cycle_info["recorded_vehicle_ids"].add(veh_id)
                    cycle_info["stop_times"].append(stop_time_after_red)

    # ==== メインループ ====
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        sim_time = traci.simulation.getTime()

        # ==== サイクル長の時間帯別切替 ====
        # 目標サイクル長が変わった時だけ、全対象信号のプログラムを差し替える。
        # 現在フェーズを引き継ぐことで色の飛びを防ぐ（フェーズ番号・状態は不変）。
        target_cycle = get_target_cycle(sim_time, warmup_time)
        if target_cycle != current_applied_cycle:
            for tl_id, scaled in cycle_scaled_logics.items():
                logic = scaled.get(target_cycle)
                if logic is None:
                    continue
                try:
                    logic.currentPhaseIndex = traci.trafficlight.getPhase(tl_id)
                    traci.trafficlight.setProgramLogic(tl_id, logic)
                except Exception as e:
                    print(f"⚠ サイクル制御: {tl_id} の切替に失敗しました: {e}")
            print(f"🔄 sim_time={sim_time:.0f}s: サイクル長を {target_cycle}s に変更しました")
            current_applied_cycle = target_cycle

        # ==== 全9道路の赤サイクル追跡（赤時間3分割計測） ====
        for tl_id, configs in TRAFFIC_LIGHT_CONFIG.items():
            try:
                current_phase = traci.trafficlight.getPhase(tl_id)
                prev_phase = prev_phases_1[tl_id]
            except Exception as e:
                print(f"⚠ Error at {tl_id} - phase: {e}")
                continue

            for i, config in enumerate(configs):
                try:
                    key = (tl_id, i)
                    if key not in red_split_writers:
                        continue
                    target = get_road_target(config)

                    # 赤開始: その道路の黄が終わった瞬間
                    if (
                        sim_time >= warmup_time
                        and prev_phase == target["yellow_phase"]
                        and current_phase != target["yellow_phase"]
                    ):
                        red_cycle_counts[key] += 1
                        active_red_cycles[key] = {
                            "road_id": str(config["id"]),
                            "cycle": red_cycle_counts[key],
                            "red_start_time": sim_time,
                            "recorded_vehicle_ids": set(),
                            "stop_times": []
                        }

                    # 赤終了: その道路が青になった瞬間 → 3分割を書き出し
                    if (
                        prev_phase is not None
                        and prev_phase != target["green_phase"]
                        and current_phase == target["green_phase"]
                    ):
                        cycle_info = active_red_cycles.pop(key, None)
                        if cycle_info is not None:
                            cycle_info["red_end_time"] = sim_time
                            write_red_split(key, cycle_info)

                    # 赤中: 停車車両を記録
                    if key in active_red_cycles:
                        record_stops(key, config, sim_time)

                except Exception as e:
                    print(f"⚠ Error at {tl_id} ({config['id']}): {e}")

            prev_phases_1[tl_id] = current_phase

    # 青まで到達しなかった未完了サイクルは赤時間が確定しないため破棄
    active_red_cycles.clear()

    # ==== 終了処理 ====
    for f in red_split_files.values():
        f.close()

    traci.close()
    print(f"✅ Day {day}: 赤時間3分割計測を完了しました → {red_split_dir_day}")

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


def prepare_route_file_for_day(day):
    if os.path.isfile(source):
        return False

    saved_route = os.path.join(destination, f"{day}日目.rou.xml")
    if os.path.isfile(saved_route):
        shutil.copy2(saved_route, source)
        print(f"✅ Day {day}: 保存済みルートを使用します → {saved_route}")
        return True

    raise FileNotFoundError(
        f"{source} が見つかりません。車両生成を有効にするか、{saved_route} を用意してください。"
    )

# ======================
# ファイル統合(赤時間3分割) — 全道路・全日を1つのCSVへ
# ======================
def combine_red_split(days):
    # 道路を列、各サイクルの区間1→2→3を縦に積む。道路はサイクル数で揃えるので位置ずれしない。
    os.makedirs(red_split_merged_dir, exist_ok=True)
    final_output_path = os.path.join(red_split_merged_dir, "赤時間3分割_統合.csv")
    seg_cols = ["区間1台数", "区間2台数", "区間3台数"]

    all_blocks = []
    for day in range(1, days + 1):
        day_folder = os.path.join(red_split_output_dir, f"day{day}")
        if not os.path.isdir(day_folder):
            continue
        # まず当日の各道路データを読み込む
        road_dfs = {}
        for road_id in ROAD_PHASE_TARGETS:  # "1".."9" の順
            file_path = os.path.join(day_folder, f"道路{road_id}_day{day}.csv")
            if not os.path.isfile(file_path):
                continue
            try:
                df = pd.read_csv(file_path, encoding='utf-8-sig')
            except Exception as e:
                print(f"❌ 読み込み失敗: {file_path} - {e}")
                continue
            if df.empty or not all(c in df.columns for c in seg_cols):
                continue
            road_dfs[road_id] = df
        if not road_dfs:
            continue
        # 道路ごとにサイクル数が違う「はみ出し」は、最小サイクル数に合わせて
        # 完全なサイクル単位で切り詰める（区間2などの途中では切らない）
        min_cycles = min(len(df) for df in road_dfs.values())
        road_series = {}  # 道路名 -> (サイクル数, 区間番号) を index にした台数
        for road_id, df in road_dfs.items():
            df = df.iloc[:min_cycles]  # 先頭 min_cycles サイクル（各サイクルは区間1/2/3が揃う）
            # 各サイクルの 区間1→区間2→区間3 を縦に並べる
            long = df.melt(id_vars=["サイクル数"], value_vars=seg_cols,
                           var_name="区間", value_name="台数")
            long["区間番号"] = long["区間"].str.extract(r"区間(\d)").astype(int)
            road_series[f"道路{road_id}"] = (
                long.set_index(["サイクル数", "区間番号"])["台数"].sort_index()
            )
        # 全道路がサイクル1..min_cyclesで揃うので欠け(NaN)は生じない
        day_df = pd.concat(road_series, axis=1).sort_index().reset_index(drop=True)
        all_blocks.append(day_df)

    if all_blocks:
        final_df = pd.concat(all_blocks, axis=0, ignore_index=True)
        # 列を 道路1..道路9 の順に整える
        ordered = [f"道路{r}" for r in ROAD_PHASE_TARGETS if f"道路{r}" in final_df.columns]
        final_df = final_df[ordered].astype("Int64")
        final_df.to_csv(final_output_path, index=False, encoding='utf-8-sig')
        print(f"\n✅ 統合完了: {final_output_path} ({len(final_df)}行 × {len(final_df.columns)}列)")
    else:
        print("⚠ 統合するデータがありませんでした。")

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
    # VEHICLE_GENERATION_RANGE = range(1, 25)  # #10 〜 #22
    days = 181

    # ======================================================
    # チェックポイント再開設定
    # True : 途中で中断した場合、最後に完了した日の翌日から再開
    # False: 常に Day 1 から開始
    # ======================================================
    RESUME_FROM_CHECKPOINT = True

    start_day = 1
    if RESUME_FROM_CHECKPOINT:
        last_completed = 0
        for d in range(1, days + 1):
            marker = os.path.join(
                red_split_output_dir, f"day{d}", f"道路1_day{d}.csv"
            )
            if os.path.isfile(marker):
                try:
                    marker_df = pd.read_csv(marker, encoding='utf-8-sig')
                    completed = len(marker_df) > 0
                except Exception:
                    completed = False

                if not completed:
                    break
                last_completed = d
            else:
                break
        if last_completed > 0:
            start_day = last_completed + 1
            print(f"✅ チェックポイント検出: Day {last_completed} まで完了。Day {start_day} から再開します。")
        else:
            print("チェックポイントなし。Day 1 から開始します。")

    for day in range(start_day, days + 1):
        saved_route = os.path.join(destination, f"{day}日目.rou.xml")
        vehicle_data_exists = os.path.isfile(saved_route)

        if vehicle_data_exists:
            # 保存済み車両データがある場合は車両生成をスキップし、シミュレーションから開始する
            print(f"\n=== Day {day}: 保存済み車両データを検出 → 車両生成をスキップ ({saved_route}) ===")
        else:
            print(f"\n=== Day {day}: 車両生成開始 ===")
            generate_vehicle_routes(time_settings, day, VEHICLE_GENERATION_RANGE) #　車両生成

        print(f"=== Day {day}: シミュレーション開始 ===")
        route_restored_from_saved = prepare_route_file_for_day(day)
        run_simulation_with_signal_logging(day, warmup_time=warmup_time) #シミュレーション

        print(f"=== Day {day} を移動開始 ===")
        if route_restored_from_saved:
            if os.path.isfile(source):
                os.remove(source)
            print(f"✅ Day {day}: 保存済みルートで実行したため、一時ルートを削除しました")
        else:
            move_and_rename_rou_file_with_day(source, destination, day) #rouファイル移動 & リネーム

        print(f"=== Day {day}: 完了 ===")

    print("\n=== 全日完了：3分割データを統合します ===")
    combine_red_split(days)  # 全道路・全日を1つのCSVに統合
