r"""
時間帯別（signal_delay方式）で ギャップ感応 vs 予測ギャップ(2s) を比較する。

各方式フォルダの
  時間帯別_signal方式_遅れ待ち_青サイクル.csv
を読み、時間帯(0〜23時)ごとに
  ・J交差点   遅れ・待ち台数（gap / 予測2s / 差）
  ・街全体     遅れ・待ち台数（gap / 予測2s / 差）
  ・Jサイクル数・平均青時間（主道路=道路1 / 従道路=道路5）
をまとめた比較CSVを出力する。

出力（BASE_OUT 直下）:
  時間帯別_signal方式_gap比較_遅れ待ち.csv   … J/街全体の遅れ・待ち（gap/予測2s/差）
  時間帯別_signal方式_gap比較_青サイクル.csv … サイクル数・平均青（両方式）

使い方: python ★時間帯別_signal方式_gap比較.py
"""
import os
import sys
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
BASE_OUT = r"C:\Users\Tsukasa\Desktop\研究\制御\csv掃き出し"
FILE = "時間帯別_signal方式_遅れ待ち_青サイクル.csv"

# (フォルダ, ラベル)  ※提案手法=予測ギャップ2s
GAP_FOLDER  = "ギャップ感応制御"
PGAP_FOLDER = "予測ギャップ制御"   # GAPSHORT=2（最良）


def _load(folder):
    path = os.path.join(BASE_OUT, folder, FILE)
    if not os.path.isfile(path):
        print(f"⚠️ 見つかりません（未実行）: {path}")
        return None
    df = pd.read_csv(path, encoding="utf-8-sig")
    # 「合計/平均」行を除外し、時間帯行だけを time(0-23) index に
    df = df[df["時"].astype(str).str.endswith("時")].copy()
    df["h"] = df["時"].str.replace("時", "", regex=False).astype(int)
    return df.set_index("h")


def _diff_cols(g, p, col):
    gv = int(g[col]); pv = int(p[col])
    d = pv - gv
    pct = (d / gv * 100) if gv else 0.0
    return gv, pv, d, round(pct, 1)


def main():
    g = _load(GAP_FOLDER)
    p = _load(PGAP_FOLDER)
    if g is None or p is None:
        print("両方式の集計が揃っていません。先にシミュレーションを実行してください。")
        return

    hours = sorted(set(g.index) | set(p.index))

    # ---- ① 遅れ・待ち（J / 街全体）----
    rows = []
    for h in hours:
        gr = g.loc[h]; pr = p.loc[h]
        jd  = _diff_cols(gr, pr, "J遅れ(秒)")
        jq  = _diff_cols(gr, pr, "J待ち台数")
        ad  = _diff_cols(gr, pr, "街全体遅れ(秒)")
        aq  = _diff_cols(gr, pr, "街全体待ち台数")
        rows.append({
            "時": f"{h:02d}時",
            "J遅れ_gap": jd[0], "J遅れ_予測2s": jd[1], "J遅れ_差": jd[2], "J遅れ_差%": jd[3],
            "J待ち_gap": jq[0], "J待ち_予測2s": jq[1], "J待ち_差": jq[2], "J待ち_差%": jq[3],
            "街遅れ_gap": ad[0], "街遅れ_予測2s": ad[1], "街遅れ_差": ad[2], "街遅れ_差%": ad[3],
            "街待ち_gap": aq[0], "街待ち_予測2s": aq[1], "街待ち_差": aq[2], "街待ち_差%": aq[3],
        })
    df1 = pd.DataFrame(rows)
    # 合計行
    tot = {"時": "合計"}
    for base in ["J遅れ", "J待ち", "街遅れ", "街待ち"]:
        gsum = df1[f"{base}_gap"].sum(); psum = df1[f"{base}_予測2s"].sum()
        tot[f"{base}_gap"] = gsum; tot[f"{base}_予測2s"] = psum
        tot[f"{base}_差"] = psum - gsum
        tot[f"{base}_差%"] = round((psum - gsum) / gsum * 100, 1) if gsum else 0.0
    df1 = pd.concat([df1, pd.DataFrame([tot])], ignore_index=True)

    p1 = os.path.join(BASE_OUT, "時間帯別_signal方式_gap比較_遅れ待ち.csv")
    df1.to_csv(p1, index=False, encoding="utf-8-sig")

    # ---- ② サイクル数・平均青（両方式）----
    rows2 = []
    for h in hours:
        gr = g.loc[h]; pr = p.loc[h]
        rows2.append({
            "時": f"{h:02d}時",
            "Jサイクル_gap": int(gr["Jサイクル数"]), "Jサイクル_予測2s": int(pr["Jサイクル数"]),
            "主道路青_gap": gr["主道路(道1)平均青(s)"], "主道路青_予測2s": pr["主道路(道1)平均青(s)"],
            "従道路青_gap": gr["従道路(道5)平均青(s)"], "従道路青_予測2s": pr["従道路(道5)平均青(s)"],
        })
    df2 = pd.DataFrame(rows2)
    p2 = os.path.join(BASE_OUT, "時間帯別_signal方式_gap比較_青サイクル.csv")
    df2.to_csv(p2, index=False, encoding="utf-8-sig")

    # ---- コンソール要約 ----
    with pd.option_context("display.max_columns", None, "display.width", 240,
                           "display.unicode.east_asian_width", True):
        print(f"\n===== ① 遅れ・待ち（J/街全体） → {p1} =====")
        print(df1[["時", "J遅れ_gap", "J遅れ_予測2s", "J遅れ_差",
                   "街遅れ_gap", "街遅れ_予測2s", "街遅れ_差"]].to_string(index=False))
        print(f"\n===== ② サイクル数・平均青 → {p2} =====")
        print(df2.to_string(index=False))

    # 主道路青の最長/最短（予測2s基準）
    d2 = df2.copy()
    d2["主青2s"] = pd.to_numeric(d2["主道路青_予測2s"], errors="coerce")
    d2 = d2.dropna(subset=["主青2s"])
    if not d2.empty:
        hmax = d2.loc[d2["主青2s"].idxmax()]; hmin = d2.loc[d2["主青2s"].idxmin()]
        print(f"\n[予測2s 主道路青] 最長={hmax['時']} {hmax['主青2s']}s / 最短={hmin['時']} {hmin['主青2s']}s")
    print("\n✅ 時間帯別 signal方式 比較 生成完了")


if __name__ == "__main__":
    main()
