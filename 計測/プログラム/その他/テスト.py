# ヒートマップ追加
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sumolib.net import readNet
from matplotlib import colormaps

# 🔤 日本語フォント（Meiryo, IPAexGothic, Noto Sans CJK JP など）
matplotlib.rcParams['font.family'] = 'Meiryo'

# ノード→エッジIDの対応（SUMO上の道路ID）
node_to_edge_map = {
    0: ["DtoA"],
    1: ["E91", "E92", "-543210000#5"],
    2: ["E12"],
    3: ["E159", "E158"],
    4: ["E26"],
    5: ["E196", "-E197", "E195"],
    6: ["-E53", "-E35"],
    7: ["E36"],
    8: ["M-1toJ"],
}

# ノード重要度（リスト形式、0のノードは描画しない）
node_importance = np.array([
    0.1395,  # 道路1（ノード0）
    0.1503,  # 道路2（ノード1）
    0.1441,  # 道路3
    0.1464,  # 道路4
    0.0000,  # 道路5
    0.1414,  # 道路6
    0.1561,  # 道路7
    0.1589,  # 道路8
    0.1447,  # 道路9
])

# SUMOネットワークファイルの読み込み
net = readNet("C:/Users/tslab/Desktop/予測/toyama_shouwa_offset_smz.net.xml")

# 描画対象（重要度 > 0 のノード）だけ正規化
valid_indices = np.where(node_importance > 0)[0]
valid_importances = node_importance[valid_indices]
min_imp, max_imp = valid_importances.min(), valid_importances.max()
importance_norm = (node_importance - min_imp) / (max_imp - min_imp + 1e-6)

# == 赤一色の濃淡カラーマップ：薄赤 → 赤 → 濃赤 =====================
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("custom_red", ["mistyrose", "red", "darkred"])

fig, ax = plt.subplots(figsize=(12, 12))

# 🟥 全道路をまず灰色で描画
for edge in net.getEdges():
    shape = edge.getShape()
    if len(shape) >= 2:
        xs, ys = zip(*shape)
        ax.plot(xs, ys, color='lightgray', linewidth=1)

# 🔥 重要ノードのみ強調描画＆ラベル表示
for node_id, edge_ids in node_to_edge_map.items():
    if node_id >= len(node_importance):
        continue
    importance = node_importance[node_id]
    if importance <= 0:
        continue  # スキップ：重要度が0以下

    color = cmap(importance_norm[node_id])

    # 複数エッジにまたがる場合のラベル位置計算用
    all_xs, all_ys = [], []
    for edge_id in edge_ids:
        try:
            edge = net.getEdge(edge_id)
            shape = edge.getShape()
            if len(shape) >= 2:
                xs, ys = zip(*shape)
                ax.plot(xs, ys, color=color, linewidth=6)
                all_xs.extend(xs)
                all_ys.extend(ys)
        except Exception as e:
            print(f"⚠ エッジ {edge_id} に関するエラー: {e}")

    # 🏷 ラベルを中心に1つだけ表示
    if all_xs and all_ys:
        center_x = sum(all_xs) / len(all_xs)
        center_y = sum(all_ys) / len(all_ys)
        ax.text(center_x, center_y, f"道路{node_id+1}", fontsize=10,
                color='black', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.5))

# 🧭 カラーバー
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_imp, vmax=max_imp))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('ノード重要度（道路1に対する影響）', fontsize=12)

# 📝 タイトルなど
ax.set_title('道路1への影響度ヒートマップ（赤=重要）', fontsize=16, weight='bold')
ax.axis('off')
plt.tight_layout()
plt.show()
