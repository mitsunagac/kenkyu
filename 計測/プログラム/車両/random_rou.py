import traci
import random
import time
import xml.etree.ElementTree as ET

# ----------------------------
# ① net.xmlからjunctionと接続情報を読み込む
# ----------------------------
def load_junction_connections(netxml_file):
    tree = ET.parse(netxml_file)
    root = tree.getroot()

    connections_by_from_edge = {}

    for connection in root.findall("connection"):
        from_edge = connection.get("from")
        to_edge = connection.get("to")
        direction = connection.get("dir")  # "r", "l", "s" など

        if from_edge not in connections_by_from_edge:
            connections_by_from_edge[from_edge] = []

        if direction == "s":
            turn_type = "straight"
        elif direction == "r":
            turn_type = "right"
        elif direction == "l":
            turn_type = "left"
        else:
            turn_type = "unknown"

        connections_by_from_edge[from_edge].append((to_edge, turn_type))

    return connections_by_from_edge

# ----------------------------
# ② 各from_edgeに対して右左折率をランダムに設定
# ----------------------------
def generate_turn_probabilities(connections):
    probs = {}
    for from_edge, to_list in connections.items():
        turn_types = {}
        for to_edge, turn_dir in to_list:
            turn_types.setdefault(turn_dir, []).append(to_edge)

        # デフォルトの比率（調整可）
        ratios = {"straight": 0.5, "right": 0.3, "left": 0.2}
        weighted_to_edges = []
        for turn_dir, edges in turn_types.items():
            prob = ratios.get(turn_dir, 0.0)
            for e in edges:
                weighted_to_edges += [e] * int(prob * 100)

        probs[from_edge] = weighted_to_edges

    return probs

# ----------------------------
# ③ メイン処理（TraCIで車両生成とルート構築）
# ----------------------------
def run_simulation(sumo_cfg, netxml_file, output_rou=r"C:\Users\takut\Desktop\町モデル\車\dynamic_routes.rou.xml", duration=3600):
    conn = load_junction_connections(netxml_file)
    turn_probs = generate_turn_probabilities(conn)

    traci.start(["sumo", "-c", sumo_cfg])
    vehicle_routes = {}
    veh_id = 0
    step = 0

    while step < duration:
        traci.simulationStep()

        # 2秒ごとに車両を生成（最大1800台）
        if step % 2 == 0:
            available_edges = list(turn_probs.keys())
            start_edge = random.choice(available_edges)
            route = [start_edge]

            # 経路生成（最大10ステップ程度）
            current_edge = start_edge
            for _ in range(10):
                next_options = turn_probs.get(current_edge, [])
                if not next_options:
                    break
                next_edge = random.choice(next_options)
                if next_edge in route:
                    break  # ループ防止
                route.append(next_edge)
                current_edge = next_edge

            route_id = f"r{veh_id}"
            traci.route.add(route_id, route)
            traci.vehicle.add(f"veh{veh_id}", routeID=route_id, depart=step)
            vehicle_routes[f"veh{veh_id}"] = route
            veh_id += 1

        step += 1

    traci.close()
    write_routes_to_rou(vehicle_routes, output_rou)

# ----------------------------
# ④ rou.xml出力関数
# ----------------------------
def write_routes_to_rou(vehicle_routes, output_file):
    root = ET.Element("routes")
    for i, (veh_id, route) in enumerate(vehicle_routes.items()):
        route_elem = ET.SubElement(root, "route", id=f"r_{veh_id}", edges=" ".join(route))
        ET.SubElement(root, "vehicle", id=veh_id, route=f"r_{veh_id}", depart=str(i * 2))  # 2秒ごと

    # インデントと改行を入れてきれいに出力（Python 3.9以降）
    ET.indent(root, space="    ", level=0)  # 4スペースでインデント

    tree = ET.ElementTree(root)
    tree.write(output_file, encoding="utf-8", xml_declaration=True)
    print(f"[完了] ルートを {output_file} に出力しました（{len(vehicle_routes)} 台）")

# ----------------------------
# 実行部分
# ----------------------------
if __name__ == "__main__":
    netxml_file = r"C:\Users\takut\Desktop\町モデル\toyama_shouwa.net.xml"           # ← あなたのネットワークファイルに置き換えてください
    sumo_cfg = r"C:\Users\takut\Desktop\町モデル\toyama_shouwa.sumocfg"              # ← あなたの.sumocfgファイルに置き換えてください
    run_simulation(sumo_cfg, netxml_file)
