import os
import csv
import re

# signal_path = r"c:\Users\tslab\Desktop\予測\制御\csv掃き出し\現最強\制御前\各道路計測ログ\信号J\信号J_合計.csv"   # ルートディレクトリ
# OUTPUT_CSV = r"c:\Users\tslab\Desktop\予測\制御\csv掃き出し\現最強\制御前\各道路計測ログ\各方向_前半後半停車台数合計.csv"

signal_path = r"c:\Users\tslab\Desktop\予測\制御\csv掃き出し\8. 記録データ (京三会議. 2.16 )\最終修正_ギャップ感応\各道路計測ログ\信号J\信号J_合計.csv"   # ルートディレクトリ
OUTPUT_CSV = r"c:\Users\tslab\Desktop\予測\制御\csv掃き出し\8. 記録データ (京三会議. 2.16 )\最終修正_ギャップ感応\各道路計測ログ\gap_各方向_前半後半停車台数合計.csv"

# signal_path = r"c:\Users\tslab\Desktop\予測\制御\csv掃き出し\現最強\青時間制御\各道路計測ログ\信号J\信号J_合計.csv"   # ルートディレクトリ
# OUTPUT_CSV = r"c:\Users\tslab\Desktop\予測\制御\csv掃き出し\現最強\青時間制御\各道路計測ログ\提案_各方向_前半後半停車台数合計.csv"

# 道路ごとの集計
# { 道路名: [odd_sum, even_sum] }
road_sums = {}

with open(signal_path, newline="", encoding="utf-8-sig") as f:
    reader = csv.reader(f)

    header = next(reader)
    header = [h.strip() for h in header]

    # 対象となる道路列（step・合計を除外）
    road_columns = [
        (i, name) for i, name in enumerate(header)
        if name.startswith("道路")
    ]

    if not road_columns:
        raise ValueError("道路列が見つかりません")

    # 初期化
    for _, road_name in road_columns:
        road_sums[road_name] = [0, 0]

    for idx, row in enumerate(reader, start=1):
        if not row:
            continue

        for col_idx, road_name in road_columns:
            value = row[col_idx].strip()
            if value == "":
                continue

            count = int(float(value))  # 0.0 対策

            if idx % 2 == 1:
                road_sums[road_name][0] += count
            else:
                road_sums[road_name][1] += count

# ===== 出力 =====
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(["道路id", "前半停止車両", "後半停止車両"])

    for road_name, (odd, even) in road_sums.items():
        writer.writerow([road_name, odd, even])

print("完了しました:", OUTPUT_CSV)