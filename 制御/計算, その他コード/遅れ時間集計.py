import pandas as pd

# ====== TRAFFIC_LIGHT_CONFIG ======
TRAFFIC_LIGHT_CONFIG = {
    "A": [
    {"id": "1", "edges": ["DtoA"], "red": "rrrryyyyyrrrryyyyy", "green": "rrrrrrrrrrrrrrrrrr"},
    {"id": "10", "edges": ["-E70"], "red": "yyyyrrrrryyyyrrrrr", "green": "rrrrrrrrrrrrrrrrrr"},
    {"id": "11", "edges": ["E2"], "red": "yyyyrrrrryyyyrrrrr", "green": "rrrrrrrrrrrrrrrrrr"},
    {"id": "12", "edges": ["-E1"], "red": "rrrryyyyyrrrryyyyy", "green": "rrrrrrrrrrrrrrrrrr"}
    ],
    "B": [
    {"id": "13", "edges": ["-E5"], "red": "rrrryyyrrrryyy", "green": "rrrrrrrrrrrrrr"},
    {"id": "14", "edges": ["E56"], "red": "yyyyrrryyyyrrr", "green": "rrrrrrrrrrrrrr"},
    {"id": "15", "edges": ["-E2"], "red": "yyyyrrryyyyrrr", "green": "rrrrrrrrrrrrrr"},
    {"id": "16", "edges": ["E55"], "red": "rrrryyyrrrryyy", "green": "rrrrrrrrrrrrrr"}
    ],
    "C": [
    {"id": "17", "edges": ["155393727#0"], "red": "rrrryyyrrrryyy", "green": "rrrrrrrrrrrrrr"},
    {"id": "18", "edges": ["-E77"], "red": "yyyyrrryyyyrrr", "green": "rrrrrrrrrrrrrr"},
    {"id": "19", "edges": ["E76"], "red": "yyyyrrryyyyrrr", "green": "rrrrrrrrrrrrrr"},
    {"id": "20", "edges": ["155390617#0"], "red": "rrrryyyrrrryyy", "green": "rrrrrrrrrrrrrr"}
    ],
    "D": [
    {"id": "2", "edges": ["E91", "E92"], "red": "rrryyyyyrrryyyyy", "green": "rrrrrrrrrrrrrrrr"},
    {"id": "21", "edges": ["-E9"], "red": "yyyrrrrryyyrrrrr", "green": "rrrrrrrrrrrrrrrr"},
    {"id": "22", "edges": ["E4"], "red": "yyyrrrrryyyrrrrr", "green": "rrrrrrrrrrrrrrrr"},
    {"id": "23", "edges": ["-E90"], "red": "rrryyyyyrrryyyyy", "green": "rrrrrrrrrrrrrrrr"},          
    ],
    "E": [
    {"id": "3", "edges": ["E12"], "red": "rrryyyyyyyy", "green": "rrrrrrrrrrr"},
    {"id": "24", "edges": ["-E7"], "red": "yyyrrrrrrrr", "green": "rrrrrrrrrrr"},
    {"id": "25", "edges": ["543210000#5", "-E92"], "red": "rrryyyyyyyy", "green": "rrrrrrrrrrr"},          
    ],
    "F": [
    {"id": "4", "edges": ["E159", "E158"], "red": "rrryyyyyrrryyyyy", "green": "rrrrrrrrrrrrrrrr"},
    {"id": "26", "edges": ["-E123"], "red": "yyyrrrrryyyrrrrr", "green": "rrrrrrrrrrrrrrrr"},
    {"id": "27", "edges": ["E179"], "red": "yyyrrrrryyyrrrrr", "green": "rrrrrrrrrrrrrrrr"},
    {"id": "28", "edges": ["-E12"], "red": "rrryyyyyrrryyyyy", "green": "rrrrrrrrrrrrrrrr"},          
    ],
    "G": [
    {"id": "29", "edges": ["-E190"], "red": "rrrryyyrrrryyy", "green": "rrrrrrrrrrrrrr"},
    {"id": "30", "edges": ["-E186"], "red": "yyyyrrryyyyrrr", "green": "rrrrrrrrrrrrrr"},
    {"id": "31", "edges": ["E185"], "red": "yyyyrrryyyyrrr", "green": "rrrrrrrrrrrrrr"},
    {"id": "32", "edges": ["-E31"], "red": "rrrryyyrrrryyy", "green": "rrrrrrrrrrrrrr"},          
    ],
    "H": [
    {"id": "33", "edges": ["E27"], "red": "rrrryyyrrrryyy", "green": "rrrrrrrrrrrrrr"},
    {"id": "34", "edges": ["-E26"], "red": "yyyyrrryyyyrrr", "green": "rrrrrrrrrrrrrr"},
    {"id": "35", "edges": ["E187", "E186"], "red": "yyyyrrryyyyrrr", "green": "rrrrrrrrrrrrrr"},
    {"id": "36", "edges": ["E25"], "red": "rrrryyyrrrryyy", "green": "rrrrrrrrrrrrrr"},          
    ],
    "I": [
    {"id": "38", "edges": ["-E54"], "red": "rrrryyyrrrryyy", "green": "rrrrrrrrrrrrrr"},
    {"id": "37", "edges": ["-E195"], "red": "yyyyrrryyyyrrr", "green": "rrrrrrrrrrrrrr"},
    {"id": "5", "edges": ["E26"], "red": "yyyyrrryyyyrrr", "green": "rrrrrrrrrrrrrr"},
    {"id": "39", "edges": ["E181"], "red": "rrrryyyrrrryyy", "green": "rrrrrrrrrrrrrr"},          
    ],
    "J": [
    {"id": "9", "edges": ["M-1toJ"], "red": "rrrryyyyyrrrryyyyy", "green": "rrrrrrrrrrrrrrrrrr"},
    {"id": "7", "edges": ["-E53", "-E35"], "red": "yyyyrrrrryyyyrrrrr", "green": "rrrrrrrrrrrrrrrrrr"},
    {"id": "6", "edges": ["E196"], "red": "yyyyrrrrryyyyrrrrr", "green": "rrrrrrrrrrrrrrrrrr"},
    {"id": "40", "edges": ["-E158"], "red": "rrrryyyyyrrrryyyyy", "green": "rrrrrrrrrrrrrrrrrr"}
    ],
    "K": [
    {"id": "41", "edges": ["E34"], "red": "rrrrrryy", "green": "rrrrrrrr"},
    {"id": "8", "edges": ["E36"], "red": "yyyyyyrr", "green": "rrrrrrrr"},
    {"id": "42", "edges": ["E35"], "red": "yyyyyyrr", "green": "rrrrrrrr"}
    ],
    "L": [
    {"id": "43", "edges": ["-E105"], "red": "rrrryyyrrrryyy", "green": "rrrrrrrrrrrrrr"},
    {"id": "44", "edges": ["-E102"], "red": "yyyyrrryyyyrrr", "green": "rrrrrrrrrrrrrr"},
    {"id": "45", "edges": ["E104"], "red": "yyyyrrryyyyrrr", "green": "rrrrrrrrrrrrrr"},
    {"id": "46", "edges": ["E97"], "red": "rrrryyyrrrryyy", "green": "rrrrrrrrrrrrrr"}
    ],
    "M": [
    {"id": "47", "edges": ["-E154"], "red": "rrryyyyyrrryyyyy", "green": "rrrrrrrrrrrrrrrr"},
    {"id": "48", "edges": ["-E218"], "red": "yyyrrrrryyyrrrrr", "green": "rrrrrrrrrrrrrrrr"},
    {"id": "49", "edges": ["E42"], "red": "yyyrrrrryyyrrrrr", "green": "rrrrrrrrrrrrrrrr"},
    {"id": "50", "edges": ["E153"], "red": "rrryyyyyrrryyyyy", "green": "rrrrrrrrrrrrrrrr"}
    ],
    "N": [
    {"id": "51", "edges": ["-E156"], "red": "rrryyyyyrrryyyyy", "green": "rrrrrrrrrrrrrrrr"},
    {"id": "52", "edges": ["-E229"], "red": "yyyrrrrryyyrrrrr", "green": "rrrrrrrrrrrrrrrr"},
    {"id": "53", "edges": ["E259"], "red": "yyyrrrrryyyrrrrr", "green": "rrrrrrrrrrrrrrrr"},
    {"id": "54", "edges": ["E155"], "red": "rrryyyyyrrryyyyy", "green": "rrrrrrrrrrrrrrrr"}
    ],
    "O": [
    {"id": "55", "edges": ["-E0"], "red": "rrryyyyyrrryyyyy", "green": "rrrrrrrrrrrrrrrr"},
    {"id": "56", "edges": ["-E214"], "red": "yyyrrrrryyyrrrrr", "green": "rrrrrrrrrrrrrrrr"},
    {"id": "57", "edges": ["E271"], "red": "yyyrrrrryyyrrrrr", "green": "rrrrrrrrrrrrrrrr"},
    {"id": "58", "edges": ["-E152"], "red": "rrryyyyyrrryyyyy", "green": "rrrrrrrrrrrrrrrr"}
    ],
    "P": [
    {"id": "59", "edges": ["StoP"], "red": "rrryyyyyrrryyyyy", "green": "rrrrrrrrrrrrrrrr"},
    {"id": "60", "edges": ["E80"], "red": "yyyrrrrryyyrrrrr", "green": "rrrrrrrrrrrrrrrr"},
    {"id": "61", "edges": ["E15"], "red": "yyyrrrrryyyrrrrr", "green": "rrrrrrrrrrrrrrrr"},
    {"id": "62", "edges": ["E0"], "red": "rrryyyyyrrryyyyy", "green": "rrrrrrrrrrrrrrrr"}
    ],
}

# ====== 入力ファイルをここで指定 ======
input_files = [
    r"C:\Users\tslab\Desktop\予測\制御\csv掃き出し\1. 記録データ_徳島学会まで\オフセット_蓄積あり\遅れ時間\全遅れ時間ログ_主道路.csv",
    r"C:\Users\tslab\Desktop\予測\制御\csv掃き出し\1. 記録データ_徳島学会まで\オフセット_蓄積あり\遅れ時間\全遅れ時間_従道路.csv",
    # "input3.csv",   ← 追加したいときはここに追記
]

# ====== 道路ID → グループ のマッピング ======
id_to_group = {}
for group, items in TRAFFIC_LIGHT_CONFIG.items():
    for item in items:
        id_to_group[str(item["id"])] = group


# ====== CSVをまとめて読み込み ======
df_list = [pd.read_csv(f) for f in input_files]
df = pd.concat(df_list, ignore_index=True)

# ====== 全データをまとめて読み込み ======
df_list = [pd.read_csv(f) for f in input_files]
df = pd.concat(df_list, ignore_index=True)

# 道路IDごとに遅れ時間を合計
delay_sum = df.groupby("道路ID")["遅れ時間(秒)"].sum()

# ====== グループごとに合計 ======
group_sum = {}
for road_id, total_delay in delay_sum.items():
    group = id_to_group.get(str(road_id))
    if group:
        group_sum[group] = group_sum.get(group, 0) + total_delay

# ====== DataFrame化（横並び形式） ======
result_df = pd.DataFrame([group_sum], columns=sorted(group_sum.keys()))

# ====== CSVに保存 ======
result_df.to_csv(r"C:\Users\tslab\Desktop\予測\制御\csv掃き出し\1. 記録データ_徳島学会まで\オフセット_蓄積あり\遅れ時間\合計.csv", index=False)

print("✅ 集計完了！ output.csv に保存しました。")