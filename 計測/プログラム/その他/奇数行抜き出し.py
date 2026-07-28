import pandas as pd

# 入力ファイルパス
input_csv = r'C:\Users\tslab\Desktop\町モデル\output\待ち台数_遅れ時間\統合\all_days_待ち_統合.csv'
# 出力ファイルパス（奇数列だけ抽出した結果を保存）
output_csv = r'C:\Users\tslab\Desktop\町モデル\前半待ち台数データ.csv'

# ヘッダーを保持して読み込み（文字化け対策に encoding 指定）
df = pd.read_csv(input_csv, encoding='cp932')

# 偶数行だけ抽出（0, 2, 4, ... → index 0, 2, 4, ...）
df_even = df.iloc[::2]  # 0から2行おき

# 保存
df_even.to_csv(output_csv, index=False, encoding='cp932')

print(f"偶数行のみ抽出して保存しました: {output_csv}")