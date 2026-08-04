r"""
赤2分割の3方式（制御なし / ギャップ感応 / 赤2分割GNN予測制御）を比較し、
論文の表4・表5 の形式でまとめる。

各方式のフォルダにある
  信号別_待ち台数_遅れ_全24h_と_6-19時.csv
（区分= 対象交差点(J) / 他交差点(J以外) / 街全体 の行を持つ）を読み、
「実測値 ＋ 制御なしとの差(約x%)」に整形して出力する。

入力: csv掃き出し/赤2分割/{制御なし, ギャップ感応制御, 予測制御}/
出力: csv掃き出し/赤2分割/
        表4_街全体_全24h.csv        表4_街全体_6-19時.csv
        表5_対象他交差点_全24h.csv   表5_対象他交差点_6-19時.csv
        赤2分割_3方式比較_まとめ.csv （表4・表5を1枚にした一覧）

使い方: python ★赤2分割_3方式比較_表4表5生成.py
       （5. 信号制御_赤2分割.py を R2_MODE=none/gap/prediction で走らせた後に実行）
"""
import os
import sys
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

BASE_OUT    = r"C:\Users\Tsukasa\Desktop\研究\制御\csv掃き出し"
RESULT_ROOT = os.path.join(BASE_OUT, "赤2分割")
SIG_FILE    = "信号別_待ち台数_遅れ_全24h_と_6-19時.csv"

# (フォルダ名, 表示名)  ※ 比較の基準は「制御なし」
# 予測制御は x1/x2 を変えるたびに別フォルダ（予測制御_x1=…_x2=…）へ出力されるので、
# 存在するものを自動で拾って全部を比較表に並べる。
BASE_METHODS = [
    ("制御なし",         "制御なし"),
    ("ギャップ感応制御", "ギャップ感応"),
]


def discover_methods():
    """RESULT_ROOT にある 予測制御* フォルダを見つけて METHODS を組み立てる。"""
    methods = list(BASE_METHODS)
    if not os.path.isdir(RESULT_ROOT):
        return methods
    preds = sorted(f for f in os.listdir(RESULT_ROOT)
                   if f.startswith("予測制御")
                   and os.path.isdir(os.path.join(RESULT_ROOT, f)))
    for f in preds:
        if f == "予測制御":
            label = "赤2分割GNN予測制御"
        elif f.startswith("予測制御_x1="):
            label = "赤2分割GNN " + f[len("予測制御_"):]      # 例: 赤2分割GNN x1=0.9_x2=0.6
        else:
            label = "赤2分割GNN " + f[len("予測制御_"):]
        methods.append((f, label))
    return methods
BASE_LABEL = "制御なし"
WINDOWS = ["全24h", "6-19時"]
GROUPS = ["対象交差点(J)", "他交差点(J以外)", "街全体"]


def _load(folder):
    """1方式の信号別CSVから {区分: {待ち_全24h, 待ち_6-19時, 遅れ_全24h, 遅れ_6-19時}} を返す。"""
    path = os.path.join(RESULT_ROOT, folder, SIG_FILE)
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path, encoding="utf-8-sig").dropna(how="all")
    out = {}
    for _, r in df.iterrows():
        k = str(r["区分"])
        if k in GROUPS:
            out[k] = {
                "待ち_全24h":  int(r["待ち台数_全24h"]),
                "待ち_6-19時": int(r["待ち台数_6-19時"]),
                "遅れ_全24h":  int(r["遅れ時間_全24h(秒)"]),
                "遅れ_6-19時": int(r["遅れ時間_6-19時(秒)"]),
            }
    return out


def _diff_str(v, base):
    """表4/表5 と同じ「+123 (約2%)」形式。1%未満は小数1桁で出す。"""
    if base in (0, None):
        return ""
    d = v - base
    pct = abs(d) / base * 100
    p = f"約{pct:.0f}%" if pct >= 1 else f"約{pct:.1f}%"
    return f"{d:+d} ({p})"


def build(data, window, METHODS):
    q   = f"待ち_{window}"
    dly = f"遅れ_{window}"
    base = data.get(BASE_LABEL)

    # ---- 表4: 街全体 ----
    rows4 = []
    for _, label in METHODS:
        m = data.get(label)
        if not m or "街全体" not in m:
            continue
        qv, dv = m["街全体"][q], m["街全体"][dly]
        qb = base["街全体"][q]   if base else None
        db = base["街全体"][dly] if base else None
        rows4.append({
            "方式": label,
            "待ち台数[台]_実測値": qv,
            "待ち台数[台]_差": "" if label == BASE_LABEL else _diff_str(qv, qb),
            "遅れ時間[s]_実測値": dv,
            "遅れ時間[s]_差": "" if label == BASE_LABEL else _diff_str(dv, db),
        })
    df4 = pd.DataFrame(rows4)

    # ---- 表5: 対象交差点(J) / 他交差点 ----
    rows5 = []
    for _, label in METHODS:
        m = data.get(label)
        if not m or "対象交差点(J)" not in m:
            continue
        row = {"方式": label}
        for grp, tag in [("対象交差点(J)", "対象交差点"), ("他交差点(J以外)", "他交差点")]:
            qv = m[grp][q]
            qb = base[grp][q] if base else None
            row[f"待ち台数_{tag}_実測"] = qv
            row[f"待ち台数_{tag}_差"] = "" if label == BASE_LABEL else _diff_str(qv, qb)
        for grp, tag in [("対象交差点(J)", "対象交差点"), ("他交差点(J以外)", "他交差点")]:
            dv = m[grp][dly]
            db = base[grp][dly] if base else None
            row[f"遅れ時間_{tag}_実測"] = dv
            row[f"遅れ時間_{tag}_差"] = "" if label == BASE_LABEL else _diff_str(dv, db)
        rows5.append(row)
    df5 = pd.DataFrame(rows5)
    return df4, df5


def build_summary(data, METHODS):
    """表4・表5を1枚にまとめた一覧（時間窓 × 区分 × 方式）。"""
    rows = []
    base = data.get(BASE_LABEL)
    for window in WINDOWS:
        for grp in GROUPS:
            for _, label in METHODS:
                m = data.get(label)
                if not m or grp not in m:
                    continue
                qv = m[grp][f"待ち_{window}"]
                dv = m[grp][f"遅れ_{window}"]
                qb = base[grp][f"待ち_{window}"] if base else None
                db = base[grp][f"遅れ_{window}"] if base else None
                rows.append({
                    "時間窓": window,
                    "区分": grp,
                    "方式": label,
                    "待ち台数[台]": qv,
                    "待ち台数_差": "" if label == BASE_LABEL else _diff_str(qv, qb),
                    "遅れ時間[s]": dv,
                    "遅れ時間_差": "" if label == BASE_LABEL else _diff_str(dv, db),
                })
    return pd.DataFrame(rows)


def main():
    METHODS = discover_methods()
    print("比較対象:", " / ".join(label for _, label in METHODS))
    data = {}
    missing = []
    for folder, label in METHODS:
        d = _load(folder)
        if d is None:
            missing.append((label, os.path.join(RESULT_ROOT, folder, SIG_FILE)))
        else:
            data[label] = d

    if missing:
        print("⚠️ 次の方式の結果が見つかりません（未実行 or 実行途中）:")
        for label, p in missing:
            print(f"   [{label}] {p}")
    if BASE_LABEL not in data:
        print(f"❌ 基準の「{BASE_LABEL}」が無いため差分を出せません。先に R2_MODE=none を実行してください。")
        return

    os.makedirs(RESULT_ROOT, exist_ok=True)
    for window in WINDOWS:
        df4, df5 = build(data, window, METHODS)
        p4 = os.path.join(RESULT_ROOT, f"表4_街全体_{window}.csv")
        p5 = os.path.join(RESULT_ROOT, f"表5_対象他交差点_{window}.csv")
        df4.to_csv(p4, index=False, encoding="utf-8-sig")
        df5.to_csv(p5, index=False, encoding="utf-8-sig")
        print(f"\n===== {window} =====")
        print(f"[表4 街全体] → {p4}")
        with pd.option_context("display.width", 200, "display.unicode.east_asian_width", True):
            print(df4.to_string(index=False))
        print(f"\n[表5 対象/他交差点] → {p5}")
        with pd.option_context("display.width", 240, "display.unicode.east_asian_width", True):
            print(df5.to_string(index=False))

    ps = os.path.join(RESULT_ROOT, "赤2分割_3方式比較_まとめ.csv")
    build_summary(data, METHODS).to_csv(ps, index=False, encoding="utf-8-sig")
    print(f"\n[まとめ] → {ps}")
    print("\n✅ 赤2分割 3方式比較 生成完了")


if __name__ == "__main__":
    main()
