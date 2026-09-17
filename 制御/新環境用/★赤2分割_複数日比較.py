# -*- coding: utf-8 -*-
"""複数日で 制御なし / ギャップ感応 / 提案手法 を比較し、日ごとのばらつきを見る。

前提: 各日について3方式を走らせておく（R2_ROU で経路ファイル＝日を切り替え、
      R2_SUFFIX で出力フォルダを分ける）。1日目は接尾辞なしの既存結果を使う。

    R2_MODE=none|gap|prediction R2_SUFFIX=_N日目 R2_ROU=<...>N日目.rou.xml \
        python "5. 信号制御_赤2分割.py"

出力: csv掃き出し/赤2分割/複数日比較/
    表4_街全体_日別.csv / 表5_対象他交差点_日別.csv / 日別まとめ.csv / 日別比較.png
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = "Meiryo"
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = r"C:\Users\Tsukasa\Desktop\研究\制御\csv掃き出し\赤2分割"
OUT_DIR = os.path.join(ROOT, "複数日比較")
os.makedirs(OUT_DIR, exist_ok=True)

SIG = "信号別_待ち台数_遅れ_全24h_と_6-19時.csv"
DAYS = [1, 2, 3]
# (フォルダ名の基幹, 表示名)。1日目は接尾辞なし。
METHODS = [("制御なし", "制御なし"),
           ("ギャップ感応制御", "ギャップ感応"),
           ("予測制御_x1=0.99_x2=0.9", "提案手法")]
COLOR = {"ギャップ感応": "#9DC3E6", "提案手法": "#ED7D31"}


def folder(base, day):
    return base if day == 1 else f"{base}_{day}日目"


def load(day):
    """その日の3方式の集計を読む。"""
    out = {}
    for base, label in METHODS:
        path = os.path.join(ROOT, folder(base, day), SIG)
        if not os.path.exists(path):
            return None, f"未実行: {os.path.relpath(path, ROOT)}"
        df = pd.read_csv(path, encoding="utf-8-sig").dropna(how="all")
        row = {}
        for _, r in df.iterrows():
            k = str(r["区分"])
            key = {"対象交差点(J)": "J", "他交差点(J以外)": "他", "街全体": "全"}.get(k)
            if key:
                row[key + "待ち"] = int(r["待ち台数_全24h"])
                row[key + "遅れ"] = int(r["遅れ時間_全24h(秒)"])
        out[label] = row
    return out, None


def diff(v, base):
    return v - base, (v - base) / base * 100


rows = []
missing = []
for day in DAYS:
    data, err = load(day)
    if data is None:
        missing.append(f"{day}日目: {err}")
        continue
    b = data["制御なし"]
    for label in ("制御なし", "ギャップ感応", "提案手法"):
        d = data[label]
        rec = {"日": f"{day}日目", "方式": label}
        for scope in ("J", "他", "全"):
            for metric in ("待ち", "遅れ"):
                k = scope + metric
                rec[k + "_実測"] = d[k]
                if label == "制御なし":
                    rec[k + "_差"] = 0
                    rec[k + "_差%"] = 0.0
                else:
                    dv, pc = diff(d[k], b[k])
                    rec[k + "_差"] = dv
                    rec[k + "_差%"] = round(pc, 2)
        rows.append(rec)

if missing:
    print("⚠ 未完了のデータがあります:")
    for m in missing:
        print("   " + m)
if not rows:
    print("✗ 集計できるデータがありません。")
    sys.exit(0)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT_DIR, "日別まとめ.csv"), index=False, encoding="utf-8-sig")

# --- 表示 ---
LABELS = [("J待ち", "対象交差点J 待ち台数[台]"), ("J遅れ", "対象交差点J 遅れ時間[s]"),
          ("他待ち", "他交差点 待ち台数[台]"), ("他遅れ", "他交差点 遅れ時間[s]"),
          ("全待ち", "街全体 待ち台数[台]"), ("全遅れ", "街全体 遅れ時間[s]")]

for key, title in LABELS:
    print(f"\n■ {title}")
    print("  日      制御なし        ギャップ感応              提案手法")
    for day in sorted(df["日"].unique()):
        s = df[df["日"] == day].set_index("方式")
        base = s.loc["制御なし", key + "_実測"]
        line = f"  {day}  {base:>9,}"
        for label in ("ギャップ感応", "提案手法"):
            v = s.loc[label, key + "_実測"]
            dv = s.loc[label, key + "_差"]
            pc = s.loc[label, key + "_差%"]
            line += f"   {v:>9,} ({dv:+,} / {pc:+.1f}%)"
        print(line)
    # 3日平均
    m = df[df["方式"] != "制御なし"].groupby("方式")[key + "_差%"].agg(["mean", "min", "max"])
    for label in ("ギャップ感応", "提案手法"):
        r = m.loc[label]
        print(f"    → {label} 平均 {r['mean']:+.1f}%（{r['min']:+.1f}〜{r['max']:+.1f}%）")

# --- グラフ: 日ごとの改善率 ---
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, (key, title) in zip(axes.ravel(), LABELS):
    days = sorted(df["日"].unique())
    w = 0.35
    for i, label in enumerate(("ギャップ感応", "提案手法")):
        vals = [df[(df["日"] == d) & (df["方式"] == label)][key + "_差%"].iloc[0] for d in days]
        ax.bar([x + (i - 0.5) * w for x in range(len(days))], vals, w,
               label=label, color=COLOR[label], edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(range(len(days)))
    ax.set_xticklabels(days)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("制御なし比 [%]", fontsize=9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
axes[0][0].legend(fontsize=9)
fig.suptitle("複数日検証: 制御なしに対する変化率（3日分・24時間）")
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.join(OUT_DIR, "日別比較.png")
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"\n✅ 保存: {out}")
print(f"✅ 保存: {os.path.join(OUT_DIR, '日別まとめ.csv')}")
