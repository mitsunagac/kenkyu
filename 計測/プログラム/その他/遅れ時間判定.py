import os
import glob
import pandas as pd
import numpy as np  # ★切り上げ用に追加

# ===== 設定 =====
input_folder = r"C:\Users\tslab\Desktop\町モデル\output\待ち台数_比較\day1"   # ★元CSVファイルがあるフォルダ
output_folder_name = "統合.csv"                     # ★出力するフォルダ名（自由に変えてOK）
diff_folder_name = "half_value_csvs" 

output_folder = os.path.join(input_folder, output_folder_name)
diff_folder = os.path.join(input_folder, diff_folder_name)
os.makedirs(output_folder, exist_ok=True)
os.makedirs(diff_folder, exist_ok=True)

# ===== 1. 各ファイルごとに処理して保存 =====
csv_files = [f for f in glob.glob(os.path.join(input_folder, "*.csv")) if "統合" not in os.path.basename(f)]
csv_files.sort()  # ファイル順を固定しておく（例: ファイル名順）

output_files = []  # 出力したファイルパスを保存（判別結果用）

for file in csv_files:
    df = pd.read_csv(file)

    if df.shape[0] <= 1 or df.shape[1] < 2:
        print(f"スキップ: データが足りない -> {file}")
        continue

    # ヘッダーを無視して、データ部分のみ取得
    data = df.iloc[1:].copy()

    # 1列目と2列目を取得
    first_col = data.iloc[:, 0].astype(float)
    second_col = data.iloc[:, 1].astype(float)

    # ★ 2列目 - 1列目 を計算
    diff_col = second_col - first_col

    # ★ (追加) 1列目と差分を並べて保存
    diff_df = pd.DataFrame({
        '元データ_1列目': first_col,
        '差分_2列目-1列目': diff_col
    })
    base_name = os.path.basename(file)
    diff_output_file = os.path.join(diff_folder, base_name)
    diff_df.to_csv(diff_output_file, index=False, encoding="utf-8-sig")

    # ★ 判別処理：（2列目 - 1列目）と（元の1列目）を比較
    result = (diff_col > first_col).astype(int)

    # 判別結果を保存
    output_df = pd.DataFrame(result)
    output_file = os.path.join(output_folder, base_name)
    output_df.to_csv(output_file, index=False, header=False, encoding="utf-8-sig")

    output_files.append(output_file)

print("個別CSVファイルの出力が完了しました。")

# ===== 2. 判別結果ファイルを横に結合 =====
combined_df_list = []

for idx, file in enumerate(output_files, start=1):
    df = pd.read_csv(file, header=None)
    header_name = f"道路{idx}"  # 上から順番にヘッダー付け
    df.columns = [header_name]
    combined_df_list.append(df)

# 横方向に結合
if combined_df_list:
    combined_df = pd.concat(combined_df_list, axis=1)
    # 結合結果を保存
    combined_output_file = os.path.join(output_folder, "統合.csv")
    combined_df.to_csv(combined_output_file, index=False, encoding="utf-8-sig")
    print(f"横結合ファイルを出力しました -> {combined_output_file}")
else:
    print("結合対象ファイルがありませんでした。")