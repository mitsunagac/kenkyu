import csv

# 元ファイルを読み込む（1列のデータと仮定）
with open(r'C:\Users\tslab\Desktop\予測\制御\csv掃き出し\制御前_各道路計測ログ\道路10.csv', newline='', encoding='utf-8') as infile:
    reader = csv.reader(infile)
    next(reader)  # ヘッダーをスキップ
    col2_data = [row[1] for row in reader if len(row) > 1]

# 2つずつペアにして横並びに
output = [col2_data[i:i+2] for i in range(0, len(col2_data), 2)]

# 出力ファイルに保存
with open(r'C:\Users\tslab\Desktop\予測\制御\csv掃き出し\制御前_各道路計測ログ\制御前_横並び_道路10.csv', 'w', newline='', encoding='utf-8') as outfile:
    writer = csv.writer(outfile)
    writer.writerows(output)
