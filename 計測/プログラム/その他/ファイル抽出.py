import os
import shutil

# 親ディレクトリ（dayフォルダたちが入っている場所）を指定
base_dir = r"C:\Users\tslab\Desktop\町モデル\output\車両流入"  # ←必要に応じて変更

# まとめる先のフォルダ
output_dir = os.path.join(base_dir, "統合J")
os.makedirs(output_dir, exist_ok=True)

# day1〜day10まで処理
for day in range(1, 11):
    day_folder = os.path.join(base_dir, f"day{day}")
    j_csv_path = os.path.join(day_folder, "J.csv")
    
    if os.path.exists(j_csv_path):
        # 出力ファイル名に day 情報を付けて区別可能にする（例：day1_J.csv）
        output_path = os.path.join(output_dir, f"day{day}_J.csv")
        shutil.copyfile(j_csv_path, output_path)
        print(f"✅ day{day} の J.csv をコピーしました")
    else:
        print(f"⚠️ day{day} に J.csv が見つかりませんでした")

print("✔️ 完了しました！")
