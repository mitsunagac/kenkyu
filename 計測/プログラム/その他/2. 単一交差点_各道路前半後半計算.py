import os
import csv

# INPUT_CSV = r"c:\Users\tslab\Desktop\予測\制御\csv掃き出し\現最強\制御前\各道路計測ログ\信号J\信号J_合計.csv"   # ルートディレクトリ
# OUTPUT_CSV = r"c:\Users\tslab\Desktop\予測\制御\csv掃き出し\現最強\制御前\各道路計測ログ\制御対象交差点_前半後半停車台数合計.csv"

# INPUT_CSV = r"c:\Users\tslab\Desktop\予測\制御\csv掃き出し\現最強\修正後_ギャップ感応\各道路計測ログ\信号J\信号J_合計.csv"   # ルートディレクトリ
# OUTPUT_CSV = r"c:\Users\tslab\Desktop\予測\制御\csv掃き出し\現最強\修正後_ギャップ感応\各道路計測ログ\制御対象交差点_前半後半停車台数合計.csv"

INPUT_CSV = r"c:\Users\tslab\Desktop\予測\制御\csv掃き出し\現最強\青時間制御\各道路計測ログ\信号J\信号J_合計.csv"   # ルートディレクトリ
OUTPUT_CSV = r"c:\Users\tslab\Desktop\予測\制御\csv掃き出し\現最強\青時間制御\各道路計測ログ\制御対象交差点_前半後半停車台数合計.csv"

odd_sum = 0
even_sum = 0

with open(INPUT_CSV, newline="", encoding="utf-8-sig") as f:
    reader = csv.reader(f)

    header = next(reader)
    header = [h.strip() for h in header]

    # if "count" not in header:
    #     raise ValueError("count 列が見つかりません")

    # count_idx = header.index("count")

    if "合計" not in header:
        raise ValueError("合計 列が見つかりません")

    count_idx = header.index("合計")

    for idx, row in enumerate(reader, start=1):
        if not row:
            continue

        count = int(float(row[count_idx]))

        if idx % 2 == 1:
            odd_sum += count
        else:
            even_sum += count

# 結果出力（Excel文字化け防止）
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(["type", "total_count"])
    writer.writerow(["前半停止車両", odd_sum])
    writer.writerow(["後半停止車両", even_sum])

print("完了しました:", OUTPUT_CSV)