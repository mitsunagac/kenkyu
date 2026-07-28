import pandas as pd

# 元のCSVファイルを読み込む
df = pd.read_csv(r"C:\Users\tslab\Desktop\予測\制御\csv掃き出し\green_durations.csv", encoding="utf-8")

# 「種類」が「主道路」のデータを抽出
df_main = df[df["種類"] == "主道路"]

# 「種類」が「従道路」のデータを抽出
df_sub = df[df["種類"] == "従道路"]

# 別々のCSVファイルに出力
df_main.to_csv("主道路.csv", index=False, encoding="utf-8-sig")
df_sub.to_csv("従道路.csv", index=False, encoding="utf-8-sig")

print("主道路と従道路に分割して保存しました。")
