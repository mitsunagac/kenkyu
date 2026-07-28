import os
import csv
import re

# BASE_DIR = r"c:\Users\tslab\Desktop\予測\制御\csv掃き出し\現最強\制御前\各道路計測ログ"   # ルートディレクトリ
# OUTPUT_CSV = r"c:\Users\tslab\Desktop\予測\制御\csv掃き出し\現最強\制御前\各道路計測ログ\ほか交差点_前半後半停車台数合計.csv"

# BASE_DIR = r"c:\Users\tslab\Desktop\予測\制御\csv掃き出し\現最強\修正後_ギャップ感応\各道路計測ログ"   # ルートディレクトリ
# OUTPUT_CSV = r"c:\Users\tslab\Desktop\予測\制御\csv掃き出し\現最強\修正後_ギャップ感応\各道路計測ログ\gap_ほか交差点前半後半停車台数合計.csv"

BASE_DIR = r"c:\Users\tslab\Desktop\予測\制御\csv掃き出し\現最強\青時間制御\各道路計測ログ"   # ルートディレクトリ
OUTPUT_CSV = r"c:\Users\tslab\Desktop\予測\制御\csv掃き出し\現最強\青時間制御\各道路計測ログ\提案_ほか交差点前半後半停車台数合計.csv"

odd_sum = 0
even_sum = 0

for signal_dir in os.listdir(BASE_DIR):
    signal_path = os.path.join(BASE_DIR, signal_dir)

    # フォルダのみ対象 & 信号A除外
    if not os.path.isdir(signal_path):
        continue
    if signal_dir == "信号J":
        continue

    for file_name in os.listdir(signal_path):
        # 道路◯.csv のみ対象
        if not (file_name.startswith("道路") and file_name.endswith(".csv")):
            continue

        csv_path = os.path.join(signal_path, file_name)

        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)

            header = next(reader)
            header = [h.strip() for h in header]

            # 念のためヘッダー確認
            if "count" not in header:
                continue

            count_idx = header.index("count")

            for idx, row in enumerate(reader, start=1):
                if not row:
                    continue

                count = int(row[count_idx])

                if idx % 2 == 1:
                    odd_sum += count
                else:
                    even_sum += count

# 結果出力
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["type", "total_count"])
    writer.writerow(["odd", odd_sum])
    writer.writerow(["even", even_sum])

print("完了しました:", OUTPUT_CSV)