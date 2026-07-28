import pandas as pd
import numpy as np

# CSVファイルを読み込み（文字化け対応でencoding指定）
df = pd.read_csv(r"C:\Users\tslab\Desktop\予測\制御\csv掃き出し\予測比較.csv", encoding="shift_jis")

# 3列目と4列目を自動で抽出（列番号指定）
col1 = df.iloc[:, 2]
col2 = df.iloc[:, 3]

# RMSEの計算
rmse = np.sqrt(((col1 - col2) ** 2).mean())

print(f"RMSE: {rmse:.4f}")
