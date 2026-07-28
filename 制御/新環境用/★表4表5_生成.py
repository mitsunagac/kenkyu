r"""
表4（街全体）・表5（対象交差点/他交差点）形式の比較CSVを生成する。

各制御方式のフォルダにある
  信号別_待ち台数_遅れ_全24h_と_6-19時.csv
（区分= 対象交差点(J) / 他交差点(J以外) / 街全体 の行を持つ）を読み、
論文の表4・表5の形（実測値＋差(約x%)）に整形して出力する。

出力（BASE_OUT 直下）:
  表4_街全体_全24h.csv, 表4_街全体_6-19時.csv
  表5_対象他交差点_全24h.csv, 表5_対象他交差点_6-19時.csv

使い方: python ★表4表5_生成.py
"""
import os
import sys
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
BASE_OUT = r"C:\Users\Tsukasa\Desktop\研究\制御\csv掃き出し"
SIG_FILE = "信号別_待ち台数_遅れ_全24h_と_6-19時.csv"

# (フォルダ名, 表示名)  ※ 提案手法 = 予測ギャップ
METHODS = [
    ("制御なし",         "制御なし"),
    ("ギャップ感応制御", "ギャップ感応"),
    ("予測ギャップ制御", "提案手法(予測ギャップ)"),
    ("予測制御_案1",     "案1"),
    ("予測制御_案2A",    "案2A"),
    ("予測制御_案2B",    "案2B"),
    ("予測制御_案3",     "案3"),
]
BASE_LABEL = "制御なし"
WINDOWS = ["全24h", "6-19時"]


def _load(folder):
    """1方式の信号別CSVから {区分: {待ち_全24h, 待ち_6-19時, 遅れ_全24h, 遅れ_6-19時}} を返す。"""
    path = os.path.join(BASE_OUT, folder, SIG_FILE)
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path, encoding="utf-8-sig").dropna(how="all")
    out = {}
    for _, r in df.iterrows():
        k = str(r["区分"])
        if k in ("対象交差点(J)", "他交差点(J以外)", "街全体"):
            out[k] = {
                "待ち_全24h": int(r["待ち台数_全24h"]),
                "待ち_6-19時": int(r["待ち台数_6-19時"]),
                "遅れ_全24h": int(r["遅れ時間_全24h(秒)"]),
                "遅れ_6-19時": int(r["遅れ時間_6-19時(秒)"]),
            }
    return out


def _diff_str(v, base):
    if base in (0, None):
        return ""
    d = v - base
    pct = abs(d) / base * 100
    p = f"約{pct:.0f}%" if pct >= 1 else f"約{pct:.1f}%"
    return f"{d:+d} ({p})"


def build(data, window):
    q = f"待ち_{window}"   # 待ち台数キー
    dly = f"遅れ_{window}"  # 遅れ時間キー
    base = data.get(BASE_LABEL)

    # ---- 表4: 街全体 ----
    rows4 = []
    for _, label in METHODS:
        m = data.get(label)
        if not m or "街全体" not in m:
            continue
        qv = m["街全体"][q]; dv = m["街全体"][dly]
        qb = base["街全体"][q] if base else None
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
            qv = m[grp][q]; dv = m[grp][dly]
            qb = base[grp][q] if base else None
            db = base[grp][dly] if base else None
            row[f"待ち台数_{tag}_実測"] = qv
            row[f"待ち台数_{tag}_差"] = "" if label == BASE_LABEL else _diff_str(qv, qb)
        for grp, tag in [("対象交差点(J)", "対象交差点"), ("他交差点(J以外)", "他交差点")]:
            dv = m[grp][dly]; db = base[grp][dly] if base else None
            row[f"遅れ時間_{tag}_実測"] = dv
            row[f"遅れ時間_{tag}_差"] = "" if label == BASE_LABEL else _diff_str(dv, db)
        rows5.append(row)
    df5 = pd.DataFrame(rows5)
    return df4, df5


def main():
    data = {}
    for folder, label in METHODS:
        d = _load(folder)
        if d is None:
            print(f"⚠️ [{label}] {SIG_FILE} が見つかりません（未実行）: {folder}")
        else:
            data[label] = d

    for window in WINDOWS:
        df4, df5 = build(data, window)
        p4 = os.path.join(BASE_OUT, f"表4_街全体_{window}.csv")
        p5 = os.path.join(BASE_OUT, f"表5_対象他交差点_{window}.csv")
        df4.to_csv(p4, index=False, encoding="utf-8-sig")
        df5.to_csv(p5, index=False, encoding="utf-8-sig")
        print(f"\n===== {window} =====")
        print(f"[表4 街全体] → {p4}")
        with pd.option_context("display.width", 200, "display.unicode.east_asian_width", True):
            print(df4.to_string(index=False))
        print(f"\n[表5 対象/他交差点] → {p5}")
        with pd.option_context("display.width", 240, "display.unicode.east_asian_width", True):
            print(df5.to_string(index=False))
    print("\n✅ 表4・表5 生成完了")


if __name__ == "__main__":
    main()
