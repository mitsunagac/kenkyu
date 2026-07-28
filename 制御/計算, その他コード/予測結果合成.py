import pandas as pd

# CSVファイルの読み込み
df = pd.read_csv(r"C:\Users\tslab\Desktop\予測\制御\csv掃き出し\予測結果_道路1.csv")

# 「前半」「後半」と「予測前半」「予測後半」を交互に並べる
rows = []
for _, row in df.iterrows():
    # 前半と予測前半
    rows.append({
        "回数": row["回数"],
        "step": row["step"],
        "実測": row["前半"],
        "予測": row["予測前半"]
    })
    # 後半と予測後半
    rows.append({
        "回数": row["回数"],
        "step": row["step"],
        "実測": row["後半"],
        "予測": row["予測後半"]
    })

# 新しいDataFrameに変換
df_long = pd.DataFrame(rows)

# 結果を保存
df_long.to_csv(r"C:\Users\tslab\Desktop\予測\制御\csv掃き出し\予測比較.csv", index=False)
