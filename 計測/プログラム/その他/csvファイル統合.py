import pandas as pd
import glob
import os

# 入力フォルダのパス
input_folder = r"C:\Users\tslab\Desktop\町モデル\output\交通環境調査\車両流入"
# input_folder = r"C:\Users\tslab\Desktop\町モデル\output\待ち台数\day1"
# 出力フォルダのパス
output_folder = r"C:\Users\tslab\Desktop\町モデル\output\交通環境調査\車両流入"
# 出力ファイル名
output_filename = "統合.csv"

# 出力フォルダが存在しない場合は作成
os.makedirs(output_folder, exist_ok=True)

# 入力フォルダ内のCSVファイルパスを取得
csv_files = glob.glob(os.path.join(input_folder, "*.csv"))

# 各CSVの2列目だけを抽出してリストに格納
df_list = []
for file in csv_files:
    df = pd.read_csv(file)
    # 2列目だけを取得（列番号1）
    second_col = df.iloc[:, [1]]
    # ファイル名を列名に設定（わかりやすくするため）
    second_col.columns = [os.path.basename(file)]
    df_list.append(second_col)

# 横方向に結合
combined_df = pd.concat(df_list, axis=1)

# 出力パスを結合してCSVとして保存
output_path = os.path.join(output_folder, output_filename)
combined_df.to_csv(output_path, index=False)

print(f"CSVファイルを保存しました: {output_path}")
