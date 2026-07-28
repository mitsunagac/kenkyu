import sumolib
import random

# === 設定 ===
NET_FILE = r"C:\Users\takut\Desktop\町モデル\toyama_shouwa.net.xml"             # 使用するネットワークファイル
ROU_FILE = r"C:\Users\takut\Desktop\町モデル\車\rou2.rou.xml"  # 出力ファイル
DEPART_INTERVAL = 2                 # 2秒ごとに1台出発
SIM_END_TIME = 46800                 # シミュレーション終了時間（秒）
MAX_ROUTE_LENGTH = 20               # ルートの最大長

# === 右左折率の設定 ===
DEFAULT_TURN_PROB = {
    "left": 0.25,
    "right": 0.25,
    "straight": 0.5
}

# ※ここにカスタム交差点のIDと進行確率を記述（必要に応じて追加）
CUSTOM_TURN_PROB = {
    "J": {
        "left": 0.1,
        "right": 0.1,
        "straight": 0.8
    },
    "E": {
        "right": 0.2,
        "straight": 0.8
    }
}

# === ネットワーク読み込み ===
net = sumolib.net.readNet(NET_FILE)

# 幹線 / 支線を分類
main_edges = []
side_edges = []

for edge in net.getEdges():
    if edge.allows("passenger") and not edge.isSpecial():
        if len(edge.getLanes()) >= 2:
            main_edges.append(edge.getID())
        else:
            side_edges.append(edge.getID())

# === 進行方向確率を取得 ===
def get_turn_prob(junction_id):
    return CUSTOM_TURN_PROB.get(junction_id, DEFAULT_TURN_PROB)

# === 交差点から次エッジを選択（右左折率を考慮） ===
def get_random_outgoing_edge(edge_id, allow_list=None):
    edge = net.getEdge(edge_id)
    from_junction = edge.getToNode().getID()
    conns = edge.getOutgoing()
    candidates = []

    turn_prob = get_turn_prob(from_junction)

    for conn in conns:
        to_edge = conn.getTo()
        if to_edge.isSpecial():
            continue
        eid = to_edge.getID()
        if allow_list is not None and eid not in allow_list:
            continue

        direction = conn.getDirection()  # "l", "r", "s", etc.
        if direction == "l":
            prob = turn_prob["left"]
        elif direction == "r":
            prob = turn_prob["right"]
        else:
            prob = turn_prob["straight"]

        candidates.append((eid, prob))

    if not candidates:
        return None

    edges, probs = zip(*candidates)
    return random.choices(edges, weights=probs)[0]

# === 支線→幹線→幹線ルート生成 ===
def generate_route_via_main(start_edge):
    route = [start_edge]
    current = start_edge
    reached_main = current in main_edges

    while len(route) < MAX_ROUTE_LENGTH:
        if reached_main:
            next_edge = get_random_outgoing_edge(current, allow_list=main_edges)
        else:
            next_edge = get_random_outgoing_edge(current)
            if next_edge in main_edges:
                reached_main = True
        if not next_edge or next_edge in route:
            break
        route.append(next_edge)
        current = next_edge

    return route if reached_main else None

# === rou.xml 出力 ===
with open(ROU_FILE, "w") as f:
    f.write('<routes>\n')
    f.write('  <vType id="car" accel="1.0" decel="4.5" sigma="0.5" length="5" maxSpeed="13.9" guiShape="passenger"/>\n')

    vehicle_id = 0
    current_time = 0
    max_attempts = 10

    while current_time < SIM_END_TIME:
        start_edge = random.choice(side_edges)
        route = None
        for _ in range(max_attempts):
            route = generate_route_via_main(start_edge)
            if route:
                break
        if route:
            f.write(f'  <vehicle id="veh{vehicle_id}" type="car" depart="{current_time}">\n')
            f.write(f'    <route edges="{" ".join(route)}"/>\n')
            f.write(f'  </vehicle>\n')
            vehicle_id += 1
            current_time += DEPART_INTERVAL
        else:
            print(f"⚠️ ルート生成失敗: {start_edge}")

    f.write('</routes>\n')

print(f"✅ 完了: {vehicle_id}台の車両が {ROU_FILE} に出力されました。")
