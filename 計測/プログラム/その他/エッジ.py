import sumolib
import csv
import os

# 読み込むネットワークファイルと出力先フォルダ
net_file = r"C:\Users\tslab\Desktop\町モデル\shin_11.12shuusei.net.xml"  # ← ここを自分のnet.xmlファイルに変更
output_dir = r"C:\Users\tslab\Desktop\町モデル\output\交通環境調査\エッジ数"      # ← 出力フォルダを好きな場所に設定

# フォルダ作成
os.makedirs(output_dir, exist_ok=True)

# ネットワーク読み込み
net = sumolib.net.readNet(net_file)

# エッジ分類
edges_lane_1 = []
edges_lane_2_or_more = []

for edge in net.getEdges():
    if edge.isSpecial():
        continue
    num_lanes = edge.getLaneNumber()
    edge_id = edge.getID()

    if num_lanes == 1:
        edges_lane_1.append(edge_id)
    elif num_lanes >= 2:
        edges_lane_2_or_more.append(edge_id)

# CSV出力（文字列として書き込む）
def write_csv(file_path, edge_ids):
    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["edge_id"])
        for eid in edge_ids:
            writer.writerow([f"'{eid}'"])  # ← ここがポイント！

write_csv(os.path.join(output_dir, "edges_with_1_lane2.csv"), edges_lane_1)
write_csv(os.path.join(output_dir, "edges_with_2_or_more_lanes2.csv"), edges_lane_2_or_more)

print(f"CSVを '{output_dir}' に文字列形式で出力しました。")