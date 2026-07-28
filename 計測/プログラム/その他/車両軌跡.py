import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from collections import defaultdict
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

#日本語対応
plt.rcParams['font.family'] = 'Meiryo'

# === ファイルパス ===
rou_file = r"C:\Users\tslab\Desktop\町モデル\random_trips_with_highway.rou.xml"
net_file = r"C:\Users\tslab\Desktop\町モデル\toyama_shouwa_no_offset.net.xml"

# === ★ 対象の時間帯（秒単位）★ ===
start_time = 1300   # 何秒以降 0時は1300 13時は48100
end_time =  90000    # 何秒まで 24時は87773

# === 1. rou.xmlのルート定義を辞書に格納 ===
route_definitions = {}
rou_tree = ET.parse(rou_file)
rou_root = rou_tree.getroot()

for route in rou_root.findall(".//route"):
    route_id = route.get("id")
    edges = route.get("edges")
    if route_id and edges:
        edge_list = edges.strip().split()
        route_definitions[route_id] = edge_list

# === 2. 指定時間帯の車両だけ抽出・通過エッジをカウント ===
edge_counts = defaultdict(int)

for vehicle in rou_root.findall(".//vehicle"):
    depart_time = float(vehicle.get("depart", "0"))
    if start_time <= depart_time <= end_time:
        route_ref = vehicle.get("route")
        if route_ref and route_ref in route_definitions:
            for edge_id in route_definitions[route_ref]:
                if not edge_id.startswith(":"):
                    edge_counts[edge_id] += 1

# === 3. net.xmlでエッジの形状を取得 ===
net_tree = ET.parse(net_file)
net_root = net_tree.getroot()

edges_geometry = {}
for edge in net_root.findall("edge"):
    edge_id = edge.get("id")
    if edge_id.startswith(":"):
        continue
    for lane in edge.findall("lane"):
        shape = lane.get("shape")
        if shape:
            coords = []
            for point in shape.strip().split(" "):
                x, y = map(float, point.split(","))
                coords.append((x, y))
            edges_geometry[edge_id] = coords

# === 4. 線分と通過回数をリスト化 ===
segments = []
weights = []

for edge_id, shape in edges_geometry.items():
    count = edge_counts.get(edge_id, 0)
    segments.append(shape)
    weights.append(count)

weights_np = np.array(weights)

# スケーリング：対数＋上限カット
clip_value = np.percentile(weights_np, 95)
clipped = np.clip(weights_np, 0, clip_value)
log_weights = np.log1p(clipped)
norm_weights = (log_weights - log_weights.min()) / (log_weights.max() - log_weights.min() + 1e-5)

# カラーマップ（白→オレンジ→赤）
colors = [
    (1.0, 1.0, 1.0),     # 白
    (1.0, 0.8, 0.3),     # 薄オレンジ
    (1.0, 0.0, 0.0),     # 赤
]
custom_cmap = LinearSegmentedColormap.from_list("white_orange_red", colors, N=256)

# === 5. 描画（黒枠＋色線）+ 台数表示のカラーバー ===
fig, ax = plt.subplots(figsize=(12, 12))

# 黒アウトライン
outline_lc = LineCollection(segments, colors="black", linewidths=3, zorder=1)
ax.add_collection(outline_lc)

# 通過回数に応じた色
lc = LineCollection(segments, cmap=custom_cmap, linewidths=2, array=norm_weights, zorder=2)
ax.add_collection(lc)

# カラーバー：台数で表示
cbar = plt.colorbar(lc, ax=ax)
cbar.set_label("通過車両台数")
tick_vals = [0.0, 0.25, 0.5, 0.75, 1.0]
tick_labels = []
for t in tick_vals:
    log_val = t * (log_weights.max() - log_weights.min()) + log_weights.min()
    original_val = np.expm1(log_val)
    tick_labels.append(f"{int(round(original_val))}台")
cbar.set_ticks(tick_vals)
cbar.set_ticklabels(tick_labels)

# スッキリ表示
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

plt.title(f"車両の通過エッジ")
plt.axis("auto")
plt.show()