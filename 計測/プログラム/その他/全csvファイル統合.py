import pandas as pd
import glob
import os

# 入力フォルダ
# input_base_folder = r"C:\Users\tslab\Desktop\町モデル\output\待ち台数"
input_base_folder = r"C:\Users\tslab\Desktop\町モデル\output\交通環境調査\車両流入"

# 出力フォルダとファイル
output_folder = os.path.join(input_base_folder, "全統合")
os.makedirs(output_folder, exist_ok=True)
final_output_path = os.path.join(output_folder, "all_days_待ち_縦統合.csv")

days = 31
vertical_df_list = []

# === 各 dayX フォルダ内のCSVを横に結合し、その結果を縦方向にためていく ===
for day in range(1, days + 1):
    input_folder = os.path.join(input_base_folder, f"day{day}")
    if not os.path.isdir(input_folder):
        print(f"スキップ: フォルダが見つかりません -> {input_folder}")
        continue

    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
    df_list = []

    for file in csv_files:
        df = pd.read_csv(file, encoding='cp932')
        if df.shape[1] < 2:
            print(f"スキップ: 列数が足りない -> {file}")
            continue
        second_col = df.iloc[:, [1]]
        df_list.append(second_col)

    if not df_list:
        print(f"スキップ: 有効なCSVがありません -> {input_folder}")
        continue

    # 横に結合してから縦用リストに追加
    combined_df = pd.concat(df_list, axis=1)
    vertical_df_list.append(combined_df)

# === 最終的にすべてを縦に結合 ===
if vertical_df_list:
    final_df = pd.concat(vertical_df_list, axis=0, ignore_index=True)

    # 任意の列名を1行目に設定
    num_columns = final_df.shape[1]
    final_headers = [f"交差点{i+1}" for i in range(num_columns)]
    final_df.columns = final_headers

    # 出力（Shift_JIS互換）
    final_df.to_csv(final_output_path, index=False, encoding='cp932')
    print(f"\n✅ 完了: 全日分を縦に統合しました -> {final_output_path}")
else:
    print("⚠ 統合するデータがありませんでした。")
