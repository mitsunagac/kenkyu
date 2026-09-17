# -*- coding: utf-8 -*-
"""走行済みの軌跡ログから、注目サイクル前後の時間距離図を切り出して保存する。

「5. 信号制御_赤2分割.py」を R2_TSWIN 付きで走らせると、各出力フォルダの
時間距離図/ に次が残る。本スクリプトはそれを読むだけなので、
シミュレーションを再実行せずに何度でも区間を変えて描き直せる。

    軌跡ログ.csv        車両軌跡（dir, sim_time, veh, edge, pos, x）
    赤信号区間.csv      交差点ごとの赤時間帯
    J毎秒_待ち遅れ.csv  サイクル特定用のJ毎秒データ
    交差点位置.csv      距離軸上の交差点位置
    green_durations.csv （1つ上の階層）Jのサイクル境界

やること:
  1. green_durations.csv からJのサイクル境界を作る
  2. J毎秒データをサイクルごとに集計し、2方式を開始時刻で突き合わせる
  3. 指定した時間帯で「差が最大／最小」のサイクルを選ぶ
  4. そのサイクルの前後3サイクル分の時間距離図を方式ごとに保存する

使い方:
    python "★赤2分割_時間距離図_サイクル抽出.py"
"""
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

matplotlib.rcParams["font.family"] = "Meiryo"
matplotlib.rcParams["axes.unicode_minus"] = False

# 実行ログから制御内容を拾う正規表現
#   ✅ [道路1制御#12] 調整[枠制限 -10→-2] pred[t3,t4]=[...] delta=10 累積=-4s
#      サイクルずれ=+7s (step=1851)
RE_CTRL = re.compile(
    r"\[道路([15])制御#(\d+)\]\s*(通常|調整)(\[枠制限[^\]]*\])?"
    r".*?delta=(-?\d+)\s+累積=(-?\d+(?:\.\d+)?)s"
    r"\s+サイクルずれ=([+-]?\d+(?:\.\d+)?)s.*?\(step=(\d+)\)")
# ギャップ感応: 🟢 主道路 青 63.0 秒 (step=1594.0)
RE_GAP_GREEN = re.compile(r"🟢 主道路 青 ([\d.]+) 秒 \(step=([\d.]+)\)")
# ギャップ感応: ✅ [GAP主道路・調整終了] G=63s, 累積=-14.0s (step=1594)
RE_GAP_ADJ = re.compile(
    r"\[GAP(主道路|従道路)・調整終了\]\s*G=([\d.]+)s,\s*累積=(-?[\d.]+)s\s*\(step=([\d.]+)\)")
C_NORMAL, C_ADJUST = "#4472C4", "#FFC000"

# True にすると時間距離図の右に「どのサイクルで何秒足したか / 調整サイクルか」の
# パネルを併設する。False なら時間距離図だけのシンプルな図になる。
# （どちらでも 制御サイクルログ.csv は出力される）
ADD_CONTROL_PANEL = False

ROOT = r"C:\Users\Tsukasa\Desktop\研究\制御\csv掃き出し\赤2分割"
PREP_TIME = 1200          # sim_time = PREP_TIME が 0時
N_AROUND = 3              # 対象サイクルの前後何サイクルを描くか

# 比較する2方式（フォルダ名, 表示名）。時間距離図の採取実行で使った接尾辞を含める。
SUFFIX = os.environ.get("R2_SUFFIX", "_時間距離図")
METHODS = [
    ("ギャップ感応制御" + SUFFIX, "ギャップ感応"),
    ("予測制御_x1=0.99_x2=0.9" + SUFFIX, "提案手法"),
]

# 注目する時間帯: (見出し, 開始時, 終了時, 指標, 選び方, 前後サイクル数, 表示のずらし)
#   worst = 提案がギャップより悪い方向に最大 / best = 良い方向に最大
#   表示のずらし … +1 で描画範囲を1サイクル後ろ（時刻が後）へずらす。-1 なら前へ。
TARGETS = [
    ("15時_待ち台数悪化", 15, 16, "J待ち台数", "worst", 2, 0),
    ("06時_遅れ時間改善", 6, 7, "J遅れ(秒)", "best", 4, 1),
]

# 横軸に描く区間（交差点位置.csv の名前）。J交差点を挟む範囲に絞ると見やすい。
X_FROM, X_TO = "M交差点", "F交差点"

OUT_DIR = os.path.join(ROOT, "時間距離図_比較")
os.makedirs(OUT_DIR, exist_ok=True)

# 時間帯別の BASE 青時間 (主, 従)。制御なしの実測と一致することを確認済み。
BASE_BY_HOUR = {}
for _h in range(24):
    if 7 <= _h < 8 or 17 <= _h < 19:
        BASE_BY_HOUR[_h] = (68.0, 34.0)      # 130s帯
    elif 8 <= _h < 17:
        BASE_BY_HOUR[_h] = (58.0, 24.0)      # 110s帯
    else:
        BASE_BY_HOUR[_h] = (53.0, 19.0)      # 100s帯


def base_green(start_s, road):
    h = int((start_s - PREP_TIME) // 3600) % 24
    return BASE_BY_HOUR[h][0 if road == "1" else 1]


def ts_dir(folder):
    return os.path.join(ROOT, folder, "時間距離図")


def load_cycles(folder):
    """green_durations.csv からJのサイクル境界 [(開始, 終了), ...] を作る。
    step は青の終了時刻なので、開始 = step - 青時間。主道路の青の開始を境界とする。"""
    path = os.path.join(ROOT, folder, "green_durations.csv")
    df = pd.read_csv(path, encoding="utf-8-sig")
    main = df[df["種類"] == "主道路"].copy()
    main["開始"] = main["step"] - main["青時間(秒)"]
    starts = sorted(main["開始"].astype(int).tolist())
    return [(starts[i], starts[i + 1]) for i in range(len(starts) - 1)]


def cycle_table(folder):
    """サイクルごとの J待ち台数・J遅れ を集計した表を返す。"""
    jsec = pd.read_csv(os.path.join(ts_dir(folder), "J毎秒_待ち遅れ.csv"),
                       encoding="utf-8-sig")
    jsec = jsec.set_index("sim_time")
    rows = []
    for s, e in load_cycles(folder):
        seg = jsec.loc[s:e - 1]
        if seg.empty:
            continue
        rows.append({"開始": s, "終了": e,
                     "J待ち台数": int(seg["J待ち台数"].sum()),
                     "J遅れ(秒)": int(seg["J遅れ(秒)"].sum())})
    return pd.DataFrame(rows)


def pair_cycles(ta, tb, tol=15):
    """2方式のサイクルを開始時刻で突き合わせる（青時間が違うため多少ずれる）。"""
    rows = []
    bi = 0
    b_starts = tb["開始"].tolist()
    for _, ra in ta.iterrows():
        while bi + 1 < len(b_starts) and b_starts[bi + 1] <= ra["開始"] + tol:
            bi += 1
        if bi >= len(b_starts):
            break
        rb = tb.iloc[bi]
        if abs(rb["開始"] - ra["開始"]) > tol:
            continue
        rows.append({"開始_A": int(ra["開始"]), "開始_B": int(rb["開始"]),
                     "待ち_A": ra["J待ち台数"], "待ち_B": rb["J待ち台数"],
                     "遅れ_A": ra["J遅れ(秒)"], "遅れ_B": rb["J遅れ(秒)"]})
    return pd.DataFrame(rows)


def parse_control_log(folder):
    """実行ログから、どのサイクルで何秒足したか / 調整サイクルかを取り出す。

    予測制御・ギャップ感応のどちらも累積補正を行うので、両方に調整サイクルがある。

    予測制御: ✅ [道路1制御#12] 調整 ... delta=10 累積=-4s サイクルずれ=+7s (step=1851)
              → 全サイクル分が残る。
    ギャップ: 🟢 主道路 青 63.0 秒 (step=1594.0)          … 主道路の全サイクル
              ✅ [GAP主道路・調整終了] G=63s, 累積=-14.0s (step=1594)  … 調整サイクルのみ
              ✅ [GAP従道路・調整終了] G=29s, 累積=-6.0s (step=2431)
              → 主道路は全サイクル、従道路は調整サイクルのみ取得できる
                （従道路の通常サイクルの青時間はログに残っていない）。
    """
    path = os.path.join(ROOT, folder, "実行ログ.txt")
    if not os.path.exists(path):
        return pd.DataFrame()

    rows = []
    gap_main = []          # (終了step, 青時間)
    gap_adj = {"1": {}, "5": {}}   # 道路 → {終了step: (G, 累積)}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = RE_CTRL.search(line)
            if m:
                road, no, tag, cap, delta, cum, cyc, step = m.groups()
                rows.append({"step": int(step), "時刻": hhmm(int(step)), "道路": road,
                             "制御回数": int(no), "サイクル種別": tag,
                             "delta(s)": float(delta), "累積(s)": float(cum),
                             "サイクルずれ(s)": float(cyc),
                             "枠制限": (cap or "").strip("[]")})
                continue
            m = RE_GAP_GREEN.search(line)
            if m:
                gap_main.append((float(m.group(2)), float(m.group(1))))
                continue
            m = RE_GAP_ADJ.search(line)
            if m:
                kind, g, cum, step = m.groups()
                gap_adj["1" if kind == "主道路" else "5"][int(float(step))] = (float(g), float(cum))

    if rows:                      # 予測制御
        return pd.DataFrame(rows).sort_values("step")

    # --- ギャップ感応 ---
    for end, dur in gap_main:     # 主道路は全サイクル
        start = end - dur
        adj = gap_adj["1"].get(int(end))
        rows.append({"step": int(start), "時刻": hhmm(start), "道路": "1",
                     "制御回数": None, "サイクル種別": "調整" if adj else "通常",
                     "delta(s)": dur - base_green(start, "1"),
                     "累積(s)": adj[1] if adj else None,
                     "サイクルずれ(s)": None, "枠制限": ""})
    for end, (g, cum) in gap_adj["5"].items():   # 従道路は調整サイクルのみ
        start = end - g
        rows.append({"step": int(start), "時刻": hhmm(start), "道路": "5",
                     "制御回数": None, "サイクル種別": "調整",
                     "delta(s)": g - base_green(start, "5"),
                     "累積(s)": cum, "サイクルずれ(s)": None, "枠制限": ""})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("step")


def hhmm(sim_time, with_sec=False):
    """sim_time[s] を時刻表記に直す（PREP_TIME が 0時0分）。"""
    t = int(sim_time) - PREP_TIME
    h, m, s = t // 3600, (t % 3600) // 60, t % 60
    return f"{h}時{m:02d}分{s:02d}秒" if with_sec else f"{h}時{m:02d}分"


_WIN_CACHE = {}


def window_bounds(folder, t):
    """t を含む記録窓の終端を返す（軌跡ログの実データから求める）。"""
    if folder not in _WIN_CACHE:
        tr = pd.read_csv(os.path.join(ts_dir(folder), "軌跡ログ.csv"),
                         encoding="utf-8-sig", usecols=["sim_time"])
        times = sorted(tr["sim_time"].unique())
        wins, s, prev = [], times[0], times[0]
        for cur in times[1:]:
            if cur - prev > 60:        # 60秒以上空いたら別の窓
                wins.append((s, prev))
                s = cur
            prev = cur
        wins.append((s, prev))
        _WIN_CACHE[folder] = wins
    for a, b in _WIN_CACHE[folder]:
        if a <= t <= b:
            return b
    return None


def draw(folder, label, t_from, t_to, out_png, title, mark, ctrl=None):
    """軌跡ログから区間を切り出して時間距離図を描く。

    ctrl（制御ログ）を渡すと、右側に時刻軸を共有した制御パネルを付ける。
    どのサイクルで何秒足したか / 調整サイクルかを図の中で突き合わせられる。
    """
    d = ts_dir(folder)
    tr = pd.read_csv(os.path.join(d, "軌跡ログ.csv"), encoding="utf-8-sig")
    tr = tr[(tr["sim_time"] >= t_from) & (tr["sim_time"] <= t_to)]
    if tr.empty:
        print(f"⚠ {label}: {t_from}〜{t_to}s に軌跡がありません")
        return None
    jp = pd.read_csv(os.path.join(d, "交差点位置.csv"), encoding="utf-8-sig")
    red = pd.read_csv(os.path.join(d, "赤信号区間.csv"), encoding="utf-8-sig")

    has_ctrl = ctrl is not None and not ctrl.empty
    fig, ax = plt.subplots(figsize=(13, 8))

    # === 時間距離図 ===
    for direction, color in (("south", "blue"), ("north", "green")):
        for _, g in tr[tr["dir"] == direction].groupby("veh"):
            g = g.sort_values("sim_time")
            ax.plot(g["x"], g["sim_time"], color=color, alpha=0.45, linewidth=0.8)

    for _, r in jp.iterrows():
        name, x = r["交差点"], r["距離(m)"]
        for _, rr in red[red["交差点"] == name].iterrows():
            s, e = max(rr["開始"], t_from), min(rr["終了"], t_to)
            if s < e:
                ax.fill_betweenx([s, e], x - 8, x + 8, color="red", alpha=0.35)

    for y in mark:
        ax.axhline(y, color="orange", linestyle="--", linewidth=1.8)

    # 横軸は対象区間（既定 N交差点〜F交差点）に絞る。全区間だと交差点が密集して読めない。
    xs = dict(zip(jp["交差点"], jp["距離(m)"]))
    x_lo, x_hi = xs[X_FROM], xs[X_TO]
    pad = (x_hi - x_lo) * 0.04
    ax.set_xlim(x_lo - pad, x_hi + pad)
    inside = jp[(jp["距離(m)"] >= x_lo) & (jp["距離(m)"] <= x_hi)]
    ax.set_xticks(inside["距離(m)"].tolist())
    ax.set_xticklabels(inside["交差点"].tolist())
    ax.set_xlabel("リンク距離 [m]")
    ax.set_ylabel("時刻")
    ax.set_ylim(t_from, t_to)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(handles=[
        plt.Line2D([0], [0], color="blue", lw=2, label="南→北"),
        plt.Line2D([0], [0], color="green", lw=2, label="北→南"),
        plt.Rectangle((0, 0), 1, 1, color="red", alpha=0.35, label="赤信号"),
        plt.Line2D([0], [0], color="orange", ls="--", lw=1.8, label="対象サイクル"),
    ], fontsize=9, loc="lower left")

    # Y軸を 〇時〇分 表記にする（1分刻み。範囲が広いときは間引く）
    span = t_to - t_from
    step = 60 if span <= 900 else (120 if span <= 1800 else 300)
    first = PREP_TIME + ((int(t_from) - PREP_TIME + step - 1) // step) * step
    ticks = list(range(first, int(t_to) + 1, step))
    ax.set_yticks(ticks)
    ax.set_yticklabels([hhmm(v) for v in ticks])

    # === 右の縦軸に「そのサイクルで何秒増減させたか / 調整サイクルか」を書く ===
    # 従道路(道路5)は、ギャップ側で通常サイクルの青時間がログに残っておらず
    # 両方式を同条件で並べられないため、主道路(道路1)だけを表示する。
    if has_ctrl:
        sub = ctrl[(ctrl["step"] >= t_from - 120) & (ctrl["step"] <= t_to)
                   & (ctrl["道路"] == "1")]
        ax2 = ax.twinx()
        ax2.set_ylim(t_from, t_to)
        pos, lab, col = [], [], []
        for _, r in sub.iterrows():
            step = int(r["step"])
            if not (t_from <= step <= t_to):
                continue
            tag = "調整" if r["サイクル種別"] == "調整" else "通常"
            pos.append(step)
            lab.append(f"{r['delta(s)']:+.0f}s（{tag}）")
            col.append(C_ADJUST if tag == "調整" else C_NORMAL)
        ax2.set_yticks(pos)
        ax2.set_yticklabels(lab, fontsize=10)
        for t, c in zip(ax2.get_yticklabels(), col):
            t.set_color(c)
        ax2.set_ylabel("主道路（道路1）青時間のBASEからの増減", fontsize=10)

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"✅ 保存: {os.path.basename(out_png)}")
    return out_png


def main():
    fa, la = METHODS[0]
    fb, lb = METHODS[1]
    for f, _ in METHODS:
        p = os.path.join(ts_dir(f), "軌跡ログ.csv")
        if not os.path.exists(p):
            print(f"✗ 軌跡ログがありません: {p}")
            print("  先に R2_TSWIN 付きでシミュレーションを実行してください。")
            return

    ta, tb = cycle_table(fa), cycle_table(fb)
    pair = pair_cycles(ta, tb)
    pair["待ち差"] = pair["待ち_B"] - pair["待ち_A"]     # 提案 - ギャップ
    pair["遅れ差"] = pair["遅れ_B"] - pair["遅れ_A"]
    pair.to_csv(os.path.join(OUT_DIR, "サイクル別_方式間差.csv"),
                index=False, encoding="utf-8-sig")
    print(f"突き合わせたサイクル数: {len(pair)}")

    # 制御ログ（どのサイクルで何秒足したか / 調整サイクルか）
    ctrl = {}
    for f, lb in METHODS:
        c = parse_control_log(f)
        ctrl[f] = c
        if not c.empty:
            c.to_csv(os.path.join(ROOT, f, "制御サイクルログ.csv"),
                     index=False, encoding="utf-8-sig")
            n = int((c["サイクル種別"] == "調整").sum())
            print(f"制御サイクルログ({lb}): 全{len(c)}件 / 調整 {n}件 ({100 * n / len(c):.1f}%)"
                  f" / delta {c['delta(s)'].min():+.0f}〜{c['delta(s)'].max():+.0f}s")
        else:
            print(f"制御サイクルログ({lb}): 記録なし")

    cycles_a = load_cycles(fa)
    for name, h0, h1, metric, how, n_around, shift in TARGETS:
        lo, hi = PREP_TIME + h0 * 3600, PREP_TIME + h1 * 3600
        sub = pair[(pair["開始_A"] >= lo) & (pair["開始_A"] < hi)]
        if sub.empty:
            print(f"⚠ {name}: 該当サイクルなし（{lo}〜{hi}s）")
            continue
        col = "待ち差" if metric == "J待ち台数" else "遅れ差"
        row = sub.loc[sub[col].idxmax()] if how == "worst" else sub.loc[sub[col].idxmin()]
        t0 = int(row["開始_A"])
        idx = min(range(len(cycles_a)), key=lambda i: abs(cycles_a[i][0] - t0))
        lo_i = max(0, idx - n_around + shift)
        hi_i = min(len(cycles_a) - 1, idx + n_around + shift)
        t_from, t_to = cycles_a[lo_i][0], cycles_a[hi_i][1]
        mark = (cycles_a[idx][0], cycles_a[idx][1])

        # 記録窓の外は軌跡が無いので、その手前で打ち切る（空白を描かない）
        avail = window_bounds(METHODS[0][0], t_from)
        note = ""
        if avail is not None and t_to > avail:
            n_after = sum(1 for i in range(idx + 1, hi_i + 1) if cycles_a[i][1] <= avail)
            note = f" ※記録窓の終端により後続は{n_after}サイクルまで"
            t_to = avail

        print(f"\n■ {name}")
        print(f"  対象サイクル: {t0}s（{hhmm(t0)}）"
              f" 待ち差 {int(row['待ち差']):+d}台 / 遅れ差 {int(row['遅れ差']):+d}秒")
        print(f"  描画範囲: {hhmm(t_from)}〜{hhmm(t_to)}（前後{n_around}サイクル）{note}")
        for folder, label in METHODS:
            draw(folder, label, t_from, t_to,
                 os.path.join(OUT_DIR, f"時間距離図_{name}_{label}.png"),
                 f"{name} / {label}（対象サイクル {hhmm(t0)}, 前後{n_around}サイクル）",
                 mark, ctrl.get(folder))


if __name__ == "__main__":
    main()
