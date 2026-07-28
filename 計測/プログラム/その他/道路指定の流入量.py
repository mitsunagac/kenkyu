import xml.etree.ElementTree as ET

def count_edges_after_depart_time(rou_file_path, target_edge_ids, depart_threshold):
    tree = ET.parse(rou_file_path)
    root = tree.getroot()

    # 各エッジIDの出現回数を記録
    edge_counts = {edge_id: 0 for edge_id in target_edge_ids}

    # すべてのルートを辞書に保持（route_id → edgeリスト）
    route_dict = {}
    for route in root.findall("route"):
        route_id = route.get("id")
        edges = route.get("edges", "").split()
        route_dict[route_id] = edges

    # 車両ごとに depart 時間を確認し、条件を満たす場合だけカウント
    for vehicle in root.findall("vehicle"):
        depart_time = float(vehicle.get("depart", "0"))
        if depart_time >= depart_threshold:
            route_id = vehicle.get("route")
            edges = route_dict.get(route_id, [])
            for edge_id in target_edge_ids:
                edge_counts[edge_id] += edges.count(edge_id)

    return edge_counts

# --- 使用例 ---
rou_path = r"C:\Users\tslab\Desktop\町モデル\random_trips_with_highway.rou.xml"
target_edges = ["M-1toJ", "E196", "-E53", "-E158"]
depart_after = 1300  # 秒数しきい値

result = count_edges_after_depart_time(rou_path, target_edges, depart_after)

# 結果を表示
for edge_id, count in result.items():
    print(f"Edge '{edge_id}' appears {count} times.")
