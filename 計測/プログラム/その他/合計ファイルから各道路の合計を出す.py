import pandas as pd

# 入力CSVを読み込み
df = pd.read_csv(r"C:\Users\tslab\Desktop\予測\制御\csv掃き出し\各道路計測ログ\信号J\信号J_合計.csv")

# 対象となる道路列（stepと合計を除外）
road_columns = [col for col in df.columns if col not in ["step", "合計"]]

results = []

for road in road_columns:
    odd_sum = df.loc[df.index % 2 == 0, road].sum()   # 奇数行（1,3,5...）
    even_sum = df.loc[df.index % 2 == 1, road].sum()  # 偶数行（2,4,6...）

    results.append({
        "道路": road,
        "前半停車台数": odd_sum,
        "後半停車台数": even_sum
    })

# 結果をDataFrameに
result_df = pd.DataFrame(results)

# CSVに出力
result_df.to_csv("J交差点_道路別合計.csv", index=False, encoding="utf-8-sig")
