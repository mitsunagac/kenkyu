r"""
赤3分割GNN制御 全案一括実行・比較レポート生成スクリプト

「4. 信号制御_赤3分割.py」を制御なし / 案1 / 案2A / 案2B / 案3（必要ならギャップ感応）で
順に実行し、各案の集計CSVを読んで比較レポートを生成する。

比較指標（全24h と 6〜19時窓 の両方）:
  ・A〜P 全信号の遅れ時間（信号別遅れ_..._6-19時.csv の「全信号合計」）
  ・J交差点(道路1/5/6/10)の待ち台数・遅れ時間（時間窓別集計_..._6-19時.csv の「全道路」）

使い方:
  python ★赤3分割_全案比較_レポート生成.py            # 全案(none,1,2A,2B,3)を24h実行
  COMPARE_END=30000 python ★赤3分割_全案比較_レポート生成.py   # デバッグ: 各案を30000秒まで
  COMPARE_GAP=1 python ★赤3分割_全案比較_レポート生成.py       # ギャップ感応も含める
  COMPARE_SKIP=1 python ★赤3分割_全案比較_レポート生成.py      # 既に結果がある案はスキップ

出力: BASE_OUT/赤3分割_全案比較_{日時}.csv
"""
import os
import sys
import subprocess
import datetime
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

HERE       = os.path.dirname(os.path.abspath(__file__))
SIM_SCRIPT = os.path.join(HERE, "4. 信号制御_赤3分割.py")
BASE_OUT   = r"C:\Users\Tsukasa\Desktop\研究\制御\csv掃き出し"

COMPARE_END  = os.environ.get("COMPARE_END", "86400")   # 各案の終了時刻(秒)。デバッグは短縮
COMPARE_GAP  = os.environ.get("COMPARE_GAP", "0") == "1"
COMPARE_SKIP = os.environ.get("COMPARE_SKIP", "0") == "1"
WIN_START, WIN_END = 6, 19   # 時間窓（制御スクリプトの ANALYSIS_START/END と合わせる）

# (mode, variant, 表示名, 出力フォルダ名)
CONFIGS = [
    ("none",       "none", "制御なし",          "制御なし"),
    ("gap",        "none", "ギャップ感応",       "ギャップ感応制御"),
    ("pgap",       "none", "予測ギャップ",       "予測ギャップ制御"),
    ("prediction", "1",    "案1(最大区間)",     "予測制御_案1"),
    ("prediction", "2A",   "案2A(前半重視)",    "予測制御_案2A"),
    ("prediction", "2B",   "案2B(後半重視)",    "予測制御_案2B"),
    ("prediction", "3",    "案3(重みA×合計中央)", "予測制御_案3"),
]
# COMPARE_GAP=0 でギャップ感応を除外可能
if not COMPARE_GAP and os.environ.get("COMPARE_GAP") == "0":
    CONFIGS = [c for c in CONFIGS if c[0] != "gap"]

WIN_LABEL = f"{WIN_START}-{WIN_END}時"


def _folder_ready(folder):
    d = os.path.join(BASE_OUT, folder)
    return os.path.isfile(os.path.join(d, "信号別_待ち台数_遅れ_全24h_と_6-19時.csv"))


def run_one(mode, variant, label, folder):
    if COMPARE_SKIP and _folder_ready(folder):
        print(f"⏭️  [{label}] 既存結果を使用（スキップ）")
        return
    print(f"\n{'='*64}\n  実行: {label}  (mode={mode}, variant={variant}, end={COMPARE_END}s)\n{'='*64}")
    env = dict(os.environ)
    env["R3_MODE"]    = mode
    env["R3_VARIANT"] = variant
    env["R3_END"]     = COMPARE_END
    env["R3_GUI"]     = "0"
    subprocess.run([sys.executable, SIM_SCRIPT], env=env, check=True)
    print(f"✅ {label} 完了")


def _read_metrics(folder):
    """1案分の指標を dict で返す。存在しなければ None。"""
    d = os.path.join(BASE_OUT, folder)
    win_path = os.path.join(d, "時間窓別集計_全24h_と_6-19時.csv")
    sig_path = os.path.join(d, "信号別遅れ_全24h_と_6-19時.csv")
    if not (os.path.isfile(win_path) and os.path.isfile(sig_path)):
        return None
    m = {}
    # J交差点(全道路) の待ち台数・遅れ（全24h / 6-19時）
    wdf = pd.read_csv(win_path, encoding="utf-8-sig")
    jrow = wdf[wdf["対象"].astype(str).str.startswith("全道路")]
    for _, r in jrow.iterrows():
        w = str(r["窓"])
        m[f"J遅れ_{w}"]   = int(r["遅れ時間合計(秒)"])
        m[f"J待ち_{w}"]   = int(r["待ち台数合計"])
        m[f"Jサイクル_{w}"] = int(r["サイクル数"])
    # A〜P 全信号の遅れ（全24h / 6-19時）
    sdf = pd.read_csv(sig_path, encoding="utf-8-sig")
    trow = sdf[sdf["信号"] == "全信号合計"]
    if not trow.empty:
        r = trow.iloc[0]
        cols = list(sdf.columns)
        m["AP遅れ_全24h"] = int(r[cols[1]])
        m["AP遅れ_" + WIN_LABEL] = int(r[cols[2]])
    return m


def generate_report():
    print(f"\n{'='*64}\n  比較レポート生成\n{'='*64}")
    data = {}
    for mode, variant, label, folder in CONFIGS:
        m = _read_metrics(folder)
        if m is None:
            print(f"⚠️ [{label}] 集計CSVが見つかりません: {folder}")
        data[label] = m or {}

    metrics = [
        ("A〜P 遅れ時間 全24h(秒)",      "AP遅れ_全24h"),
        (f"A〜P 遅れ時間 {WIN_LABEL}(秒)", "AP遅れ_" + WIN_LABEL),
        ("J交差点 遅れ時間 全24h(秒)",     "J遅れ_全24h"),
        (f"J交差点 遅れ時間 {WIN_LABEL}(秒)", "J遅れ_" + WIN_LABEL),
        ("J交差点 待ち台数 全24h",         "J待ち_全24h"),
        (f"J交差点 待ち台数 {WIN_LABEL}",    "J待ち_" + WIN_LABEL),
        (f"Jサイクル数 {WIN_LABEL}",         "Jサイクル_" + WIN_LABEL),
    ]
    labels = [c[2] for c in CONFIGS]
    base_label = "制御なし"

    columns = ["指標"]
    for lab in labels:
        columns.append(lab)
        if lab != base_label:
            columns.append(f"差({lab})")
            columns.append(f"改善率{lab}(%)")

    rows = []
    for disp, key in metrics:
        row = {"指標": disp}
        base = data.get(base_label, {}).get(key, "")
        for lab in labels:
            v = data.get(lab, {}).get(key, "")
            row[lab] = v
            if lab != base_label:
                if v != "" and base not in ("", 0):
                    row[f"差({lab})"] = v - base
                    row[f"改善率{lab}(%)"] = round((base - v) / base * 100, 1)  # 減少=正=改善
                else:
                    row[f"差({lab})"] = ""
                    row[f"改善率{lab}(%)"] = ""
        rows.append(row)

    df = pd.DataFrame(rows, columns=columns)
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    out_path = os.path.join(BASE_OUT, f"赤3分割_全案比較_{now}.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ 比較レポート出力: {out_path}\n")
    # コンソール表示（改善率つき要約）
    with pd.option_context("display.max_columns", None, "display.width", 200,
                           "display.unicode.east_asian_width", True):
        print(df.to_string(index=False))
    print("\n※ 改善率(%)は「制御なしから何%減ったか」。正=改善(遅れ減)、負=悪化。")


if __name__ == "__main__":
    print(f"▶ 比較対象: {', '.join(c[2] for c in CONFIGS)}")
    print(f"▶ 各案の終了時刻: {COMPARE_END}s / 窓: {WIN_LABEL} / スキップ: {COMPARE_SKIP}")
    for mode, variant, label, folder in CONFIGS:
        run_one(mode, variant, label, folder)
    # 表4・表5（街全体 / 対象交差点・他交差点）を生成
    subprocess.run([sys.executable, os.path.join(HERE, "★表4表5_生成.py")], check=False)
    print("\n✅ 全方式実行・表4/表5生成 完了")
