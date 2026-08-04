r"""
赤2分割GNN予測制御の x1/x2 調整用。対象交差点(J)の待ち台数・遅れ時間が
ギャップ感応を上回ったか（＝より大きく減らせたか）だけを手早く確認する。

使い方: python ★赤2分割_J評価_重み調整用.py
"""
import os
import sys
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ROOT = r"C:\Users\Tsukasa\Desktop\研究\制御\csv掃き出し\赤2分割"
SIG_FILE = "信号別_待ち台数_遅れ_全24h_と_6-19時.csv"


def load(folder):
    p = os.path.join(ROOT, folder, SIG_FILE)
    if not os.path.isfile(p):
        return None
    df = pd.read_csv(p, encoding="utf-8-sig").dropna(how="all")
    for _, r in df.iterrows():
        if str(r["区分"]) == "対象交差点(J)":
            return {"待ち": int(r["待ち台数_全24h"]), "遅れ": int(r["遅れ時間_全24h(秒)"]),
                    "待ち窓": int(r["待ち台数_6-19時"]), "遅れ窓": int(r["遅れ時間_6-19時(秒)"])}
    return None


base = load("制御なし")
gap = load("ギャップ感応制御")
if base is None or gap is None:
    print("❌ 制御なし / ギャップ感応制御 の結果がありません")
    sys.exit(1)


def pct(v, b):
    return (v - b) / b * 100


print("=== 対象交差点(J) 全24h ===")
print(f"{'方式':<26}{'待ち台数':>10}{'対制御なし':>12}{'遅れ時間':>12}{'対制御なし':>12}{'判定':>8}")
print(f"{'制御なし':<26}{base['待ち']:>10,}{'':>12}{base['遅れ']:>12,}{'':>12}")
print(f"{'ギャップ感応(目標)':<26}{gap['待ち']:>10,}{pct(gap['待ち'], base['待ち']):>+11.1f}%"
      f"{gap['遅れ']:>12,}{pct(gap['遅れ'], base['遅れ']):>+11.1f}%")
print("-" * 82)

folders = sorted(f for f in os.listdir(ROOT)
                 if f.startswith("予測制御") and os.path.isdir(os.path.join(ROOT, f)))
best = None
for f in folders:
    d = load(f)
    if d is None:
        print(f"{f:<26}{'(実行中/未完了)':>10}")
        continue
    win_q = d["待ち"] < gap["待ち"]
    win_d = d["遅れ"] < gap["遅れ"]
    mark = "★達成" if (win_q and win_d) else ("待ちのみ" if win_q else ("遅れのみ" if win_d else "未達"))
    print(f"{f:<26}{d['待ち']:>10,}{pct(d['待ち'], base['待ち']):>+11.1f}%"
          f"{d['遅れ']:>12,}{pct(d['遅れ'], base['遅れ']):>+11.1f}%{mark:>8}")
    score = (d["待ち"] - gap["待ち"]) / gap["待ち"] + (d["遅れ"] - gap["遅れ"]) / gap["遅れ"]
    if best is None or score < best[0]:
        best = (score, f, d)

if best:
    _, f, d = best
    print(f"\n現時点で最良: {f}")
    print(f"  ギャップ感応との差 … 待ち台数 {d['待ち'] - gap['待ち']:+,}台 / "
          f"遅れ時間 {d['遅れ'] - gap['遅れ']:+,}秒（どちらもマイナスなら達成）")
