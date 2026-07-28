import pandas as pd

# CSVファイルのパスを指定
file_path = r'C:\Users\tslab\Desktop\予測\新環境待ち台数データ_12.22.csv'  # ←適宜ファイル名に置き換えてください

# CSVを読み込む
df = pd.read_csv(file_path, encoding='shift_jis')

# -1 を 0 に置き換える（全データ対象）
df = df.replace(-1, 0)
df = df.replace(-2, 0)

# 上書き保存（元ファイルを上書きしたくない場合は別名にしてください）
df.to_csv(file_path, index=False)
