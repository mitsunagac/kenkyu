import random
import csv
import sumolib

# ========== 設定 ==========

netfile = r"C:\Users\takut\Desktop\町モデル\toyama_shouwa.net.xml"
to_edge_csv = r"C:\Users\takut\Desktop\町モデル\output\交通環境調査\edges_with_2_or_more_lanes.csv"
highway_edge_csv = r"C:\Users\takut\Desktop\町モデル\output\交通環境調査\edges_with_2_or_more_lanes.csv"
outputfile = r"C:\Users\takut\Desktop\町モデル\random_trips_with_highway.rou.xml"

simulation_time = 3600  # 秒（この時間内に車を生成）
depart_min = 1
depart_max = 2

north_edges = ['E1', 'E90', 'E77', 'E56']  # ← 指定された目的地エッジ（北エリア）
area_weights = [0.6, 0.4]  # 北エリア60%、その他40%

# ========== データ読み込み ==========

net = sumolib.net.readNet(netfile)
from_edges = [e.getID() for e in net.getEdges() if not e.getID().startswith(":")]

# 目的地エッジ読み込み
all_to_edges = []
with open(to_edge_csv, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        edge_id = row['edge_id'].strip().strip("'\"")
        if edge_id:
            all_to_edges.append(edge_id)

other_edges = [e for e in all_to_edges if e not in north_edges]
area_choices = [north_edges, other_edges]

# 幹線道路エッジ読み込み
highway_edges = []
with open(highway_edge_csv, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        edge_id = row['edge_id'].strip().strip("'\"")
        if edge_id:
            highway_edges.append(edge_id)

# ========== 目的地選択関数 ==========

def pick_destination(from_edge):
    for _ in range(5):
        group = random.choices(area_choices, weights=area_weights, k=1)[0]
        if not group:
            continue
        to_edge = random.choice(group)
        if to_edge != from_edge:
            return to_edge
    return None

# ========== 車両ルート生成 ==========

valid_routes = []
current_time = 0
vehicle_index = 0

print("⏳ 幹線道路経由ルートを生成中...")

while current_time <= simulation_time:
    from_edge = random.choice(from_edges)
    to_edge = pick_destination(from_edge)

    if not to_edge or from_edge == to_edge:
        continue

    try:
        from_edge_obj = net.getEdge(from_edge)
        to_edge_obj = net.getEdge(to_edge)

        # 出発地 → 最寄り幹線道路
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

        if not path_to_highway or not best_hwy or best_hwy == to_edge:
            continue

        # 幹線道路 → 目的地
        try:
            path_to_dest, _ = net.getShortestPath(net.getEdge(best_hwy), to_edge_obj)
        except:
            continue

        if not path_to_dest:
            continue

        # ✅ エッジの重複を除く
        if path_to_highway and path_to_dest:
            if path_to_highway[-1].getID() == path_to_dest[0].getID():
                path_to_dest = path_to_dest[1:]

        # 最終ルート
        final_route_edges = path_to_highway + path_to_dest
        final_route_ids = [edge.getID() for edge in final_route_edges]

        valid_routes.append((vehicle_index, current_time, final_route_ids))
        vehicle_index += 1
        current_time += random.randint(depart_min, depart_max)

    except Exception:
        continue

# ========== .rou.xml に出力 ==========

with open(outputfile, "w") as f:
    f.write("<routes>\n")
    f.write('  <vType id="car" accel="2.5" decel="4.5" maxSpeed="25" length="5" sigma="0.5"/>\n')

    for veh_id, depart_time, route_edges in valid_routes:
        route_str = " ".join(route_edges)
        f.write(f'  <route id="route{veh_id}" edges="{route_str}"/>\n')
        f.write(f'  <vehicle id="veh{veh_id}" type="car" route="route{veh_id}" depart="{depart_time}"/>\n')

    f.write("</routes>\n")

print(f"✅ 合計 {len(valid_routes)} 台の車両を {simulation_time} 秒間で生成しました。")
print(f"📄 出力ファイル：{outputfile}")
