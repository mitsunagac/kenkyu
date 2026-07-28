import os
import csv
import re

# signal_path = r"c:\Users\tslab\Desktop\予測\制御\csv掃き出し\現最強\制御前\各道路計測ログ\信号J\信号J_合計.csv"   # ルートディレクトリ
# OUTPUT_CSV = r"c:\Users\tslab\Desktop\予測\制御\csv掃き出し\現最強\制御前\各道路計測ログ\各方向_前半後半停車台数合計.csv"

signal_path = r"c:\Users\tslab\Desktop\予測\制御\csv掃き出し\現最強\修正後_ギャップ感応\各道路計測ログ\信号J\道路10.csv"   # ルートディレクトリ
OUTPUT_CSV = r"c:\Users\tslab\Desktop\予測\制御\csv掃き出し\現最強\ギャップ\現示時間\遅れ時間計算用.csv"

# signal_path = r"c:\Users\tslab\Desktop\予測\制御\csv掃き出し\現最強\青時間制御\各道路計測ログ\信号J\信号J_合計.csv"   # ルートディレクトリ
# OUTPUT_CSV = r"c:\Users\tslab\Desktop\予測\制御\csv掃き出し\現最強\青時間制御\各道路計測ログ\提案_各方向_前半後半停車台数合計.csv"

rows_out = []
current_odd = None  # 奇数行の count を一時保存

with open(signal_path, newline="", encoding="utf-8-sig") as f:
    reader = csv.reader(f)

    header = next(reader)
    header = [h.strip() for h in header]

    if "count" not in header:
        raise ValueError("count 列が見つかりません")

    count_idx = header.index("count")

    for idx, row in enumerate(reader, start=1):
        if not row:
            continue

        value = row[count_idx].strip()
        if value == "":
            continue

        count = int(float(value))  # 0.0 対策

        if idx % 2 == 1:
            # 奇数行
            current_odd = count
        else:
            # 偶数行 → セットで出力
            rows_out.append([current_odd, count])
            current_odd = None

    # 奇数行で終わった場合
    if current_odd is not None:
        rows_out.append([current_odd, ""])

# ===== 出力 =====
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(["奇数", "偶数"])
    writer.writerows(rows_out)

print("完了しました:", OUTPUT_CSV)