r"""
N単独制御 を含む4方式を signal_delay方式（標準指標）で比較する。

対象方式（フォルダ / ラベル）:
  制御なし / ギャップ感応制御(gap) / 予測ギャップ制御(予測2s) / N単独制御

各方式フォルダの 時間帯別_signal方式_遅れ待ち_青サイクル.csv を読み、
  ・総合表（制御なし基準の差つき）: J/街全体の 遅れ・待ち台数
  ・時間帯別 J遅れ・街全体遅れ（4方式並べ）
  ・時間帯別 平均青（主道路=道路1 / 従道路=道路5, 4方式並べ）
を出力する。

出力（BASE_OUT 直下）:
  N比較_総合_signal方式.csv
  N比較_時間帯別_遅れ_signal方式.csv
  N比較_時間帯別_青_signal方式.csv

使い方: python ★N単独_4方式比較.py
"""
import os
import sys
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
BASE_OUT = r"C:\Users\Tsukasa\Desktop\研究\制御\csv掃き出し"
FILE = "時間帯別_signal方式_遅れ待ち_青サイクル.csv"

METHODS = [
    ("制御なし",          "制御なし"),
    ("ギャップ感応制御",  "gap"),
    ("予測ギャップ制御",  "予測2s"),
    ("N単独制御",         "N単独"),
]
BASE_LABEL = "制御なし"


def _load(folder):
    path = os.path.join(BASE_OUT, folder, FILE)
    if not os.path.isfile(path):
        print(f"⚠️ 未実行: {path}")
        return None, None
    df = pd.read_csv(path, encoding="utf-8-sig")
    hourly = df[df["時"].astype(str).str.endswith("時")].copy()
    hourly["h"] = hourly["時"].str.replace("時", "", regex=False).astype(int)
    hourly = hourly.set_index("h")
    tot = df[df["時"].astype(str).str.startswith("合計")]
    total = tot.iloc[0] if not tot.empty else hourly.sum()
    return hourly, total


def main():
    H, T = {}, {}
    for folder, label in METHODS:
        h, t = _load(folder)
        if h is not None:
            H[label] = h; T[label] = t
    if BASE_LABEL not in T:
        print("⚠️ 制御なし基準が無いので差は出しません。")

    labels = [l for _, l in METHODS if l in T]

    # ---- ① 総合表（制御なし基準の差）----
    metrics = [("J遅れ(秒)", "J遅れ"), ("J待ち台数", "J待ち"),
               ("街全体遅れ(秒)", "街遅れ"), ("街全体待ち台数", "街待ち")]
    rows = []
    for col, disp in metrics:
        row = {"指標": disp}
        base = int(T[BASE_LABEL][col]) if BASE_LABEL in T else None
        for l in labels:
            v = int(T[l][col]); row[l] = v
            if base is not None and l != BASE_LABEL:
                d = v - base
                row[f"差({l})"] = d
                row[f"改善%({l})"] = round((base - v) / base * 100, 1) if base else 0
        rows.append(row)
    df1 = pd.DataFrame(rows)
    p1 = os.path.join(BASE_OUT, "N比較_総合_signal方式.csv")
    df1.to_csv(p1, index=False, encoding="utf-8-sig")

    # ---- ② 時間帯別 遅れ（J / 街全体）----
    hours = sorted(set().union(*[set(h.index) for h in H.values()]))
    rows2 = []
    for hh in hours:
        r = {"時": f"{hh:02d}時"}
        for l in labels:
            r[f"J_{l}"] = int(H[l].loc[hh, "J遅れ(秒)"]) if hh in H[l].index else ""
        for l in labels:
            r[f"街_{l}"] = int(H[l].loc[hh, "街全体遅れ(秒)"]) if hh in H[l].index else ""
        rows2.append(r)
    df2 = pd.DataFrame(rows2)
    p2 = os.path.join(BASE_OUT, "N比較_時間帯別_遅れ_signal方式.csv")
    df2.to_csv(p2, index=False, encoding="utf-8-sig")

    # ---- ③ 時間帯別 平均青（主道路 / 従道路）----
    rows3 = []
    for hh in hours:
        r = {"時": f"{hh:02d}時"}
        for l in labels:
            r[f"主青_{l}"] = H[l].loc[hh, "主道路(道1)平均青(s)"] if hh in H[l].index else ""
        for l in labels:
            r[f"従青_{l}"] = H[l].loc[hh, "従道路(道5)平均青(s)"] if hh in H[l].index else ""
        rows3.append(r)
    df3 = pd.DataFrame(rows3)
    p3 = os.path.join(BASE_OUT, "N比較_時間帯別_青_signal方式.csv")
    df3.to_csv(p3, index=False, encoding="utf-8-sig")

    # ---- コンソール ----
    with pd.option_context("display.max_columns", None, "display.width", 260,
                           "display.unicode.east_asian_width", True):
        print(f"\n===== ① 総合（signal方式・制御なし基準） → {p1} =====")
        print(df1.to_string(index=False))
        print(f"\n===== ② 時間帯別 遅れ（抜粋 6-19時 J/街） → {p2} =====")
        show = df2[df2["時"].isin([f"{h:02d}時" for h in range(6, 20)])]
        jcols = ["時"] + [f"J_{l}" for l in labels]
        print(show[jcols].to_string(index=False))
        print(f"\n===== ③ 時間帯別 主道路平均青（抜粋 6-19時） → {p3} =====")
        mcols = ["時"] + [f"主青_{l}" for l in labels]
        print(df3[df3["時"].isin([f"{h:02d}時" for h in range(6, 20)])][mcols].to_string(index=False))
    print("\n✅ N単独 4方式比較 生成完了")


if __name__ == "__main__":
    main()
