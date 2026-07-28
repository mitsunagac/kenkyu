import pandas as pd
import glob
import os

# 入力と出力のベースフォルダ
# input_base_folder = r"C:\Users\tslab\Desktop\町モデル\output\待ち台数"
# output_folder = r"C:\Users\tslab\Desktop\町モデル\output\待ち台数\統合"
input_base_folder = r"C:\Users\tslab\Desktop\町モデル\output\車両流入"
output_folder =r"C:\Users\tslab\Desktop\町モデル\output\車両流入\統合"

# 出力フォルダが存在しない場合は作成
os.makedirs(output_folder, exist_ok=True)

# 処理する日数（day1〜day31）
days = 31

for day in range(1, days + 1):
    input_folder = os.path.join(input_base_folder, f"day{day}")
    output_filename = f"day{day}_車両流入.csv"
    output_path = os.path.join(output_folder, output_filename)

    # 対象フォルダが存在しない場合はスキップ
    if not os.path.isdir(input_folder):
        print(f"スキップ: フォルダが見つかりません -> {input_folder}")
        continue

    # フォルダ内のCSVファイルを取得
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))

    # データフレームのリスト
    df_list = []
    for file in csv_files:
        df = pd.read_csv(file)
        if df.shape[1] < 2:
            print(f"スキップ: 列数が足りない -> {file}")
            continue
        second_col = df.iloc[:, [1]]
        second_col.columns = [os.path.basename(file)]
        df_list.append(second_col)

    if not df_list:
        print(f"スキップ: 有効なCSVがありません -> {input_folder}")
        continue

    # 横方向に結合して保存
    combined_df = pd.concat(df_list, axis=1)
    combined_df.to_csv(output_path, index=False)
    print(f"Day {day}: 統合完了 -> {output_path}")
