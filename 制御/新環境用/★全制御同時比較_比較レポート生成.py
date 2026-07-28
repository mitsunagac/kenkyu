r"""
全制御方法一括実行・比較レポート生成スクリプト

指定した制御手法をシミュレーション実行し、
完了後に3モードの比較レポートを生成する。

使い方:
  python ★全制御同時比較_比較レポート生成.py
  python ★全制御同時比較_比較レポート生成.py prediction
  python ★全制御同時比較_比較レポート生成.py none gap
  python ★全制御同時比較_比較レポート生成.py all
  python ★全制御同時比較_比較レポート生成.py prediction --route-dir C:\Users\Tsukasa\Desktop\研究\予測\町モデルデータ\生成された車

出力ファイル名: BASE_OUT/制御比較_閾値{OFFSET_ADJUST_THRESHOLD}s_{日時}.csv
"""

import sys
import os
import re
import subprocess
import datetime
import argparse
from pathlib import Path
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

# ============================================================
# 設定（BASE_OUT のみここで設定、それ以外は比較機能付きから自動取得）
# ============================================================
DEFAULT_ROUTE_DIR = r"C:\Users\Tsukasa\Desktop\研究\予測\町モデルデータ\生成された車"
# 制御スクリプト（実体ファイル名は「○ 信号制御_mix_到着間隔.py」）。
# 日本語パスの引数化で文字化けしないよう、絶対パスを直接指定する。
# 環境変数 COMPARE_SIM_SCRIPT / COMPARE_BASE_OUT で上書き可能（別制御方式の比較用）。
# ※ SIM_SCRIPT を差し替える場合、その制御スクリプト側の BASE_OUT も COMPARE_BASE_OUT と
#    同じフォルダに揃えること（出力先と読取先を一致させるため）。
SIM_SCRIPT = os.environ.get(
    "COMPARE_SIM_SCRIPT",
    r"C:\Users\Tsukasa\Desktop\研究\制御\新環境用\○ 信号制御_mix_到着間隔.py")
BASE_OUT   = os.environ.get(
    "COMPARE_BASE_OUT",
    r"C:\Users\Tsukasa\Desktop\研究\制御\csv掃き出し")


def _read_setting(key, default):
    """SIM_SCRIPT（○ 信号制御_mix_到着間隔.py）から設定値を読み取る。"""
    try:
        with open(SIM_SCRIPT, encoding="utf-8") as f:
            for line in f:
                m = re.match(rf'\s*{re.escape(key)}\s*=\s*(\S+)', line)
                if m:
                    val = m.group(1).strip('"\'')
                    try:
                        return int(val)
                    except ValueError:
                        return val
    except Exception:
        pass
    return default


OFFSET_ADJUST_THRESHOLD = _read_setting("OFFSET_ADJUST_THRESHOLD", 10)
print(f"📋 OFFSET_ADJUST_THRESHOLD = {OFFSET_ADJUST_THRESHOLD}s（○ 信号制御_mix_到着間隔.py から取得）")

# 実行順序（変更不要）
ALL_MODES = [
    ("none",       "制御なし"),
    ("gap",        "ギャップ感応制御"),
    ("prediction", "予測制御"),
]
REQUIRED_COMPARISON_MODES = [
    ("none",       "制御なし"),
    ("prediction", "予測制御"),
]
MODE_LABELS = dict(ALL_MODES)
MODE_FOLDERS = {
    "none": "制御なし",
    "gap": "ギャップ感応制御",
    "prediction": "予測制御",
}
MODE_REPORT_COLUMNS = {
    "none": "制御なし",
    "gap": "ギャップ感応",
    "prediction": "予測制御",
}
MODE_DIFF_COLUMNS = {
    "gap": "差(ギャップvs制御なし)",
    "prediction": "差(予測vs制御なし)",
}
MODE_ALIASES = {
    "none": "none",
    "制御なし": "none",
    "なし": "none",
    "no": "none",
    "gap": "gap",
    "ギャップ": "gap",
    "ギャップ感応": "gap",
    "ギャップ感応制御": "gap",
    "prediction": "prediction",
    "predict": "prediction",
    "予測": "prediction",
    "予測制御": "prediction",
    "compare": "compare",
    "比較": "compare",
}

# 比較レポート用設定
SUMMARY_FILE = "交差点別_待ち台数_遅れ時間_集計.csv"
CYCLE_FILE   = "サイクル数.csv"

SECTIONS = [
    ("町全体",      "全体合計"),
    ("J交差点のみ", "J交差点合計"),
    ("J交差点以外", "J以外合計"),
]


# ============================================================
# シミュレーション実行
# ============================================================
def _route_run_name(route_file):
    name = Path(route_file).name
    for suffix in (".rou.xml", ".xml"):
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return Path(name).stem


def _ordered_modes(mode_keys):
    wanted = set(mode_keys)
    return [(mode, label) for mode, label in ALL_MODES if mode in wanted]


def _results_exist(folder, base_out=BASE_OUT):
    """集計CSVとサイクル数CSVが両方存在するか確認する。"""
    dirpath = os.path.join(base_out, folder)
    return (os.path.exists(os.path.join(dirpath, SUMMARY_FILE)) and
            os.path.exists(os.path.join(dirpath, CYCLE_FILE)))


def run_simulation(mode, label, folder, base_out=BASE_OUT, route_file=None, skip_existing=False):
    if (skip_existing or mode == "none") and _results_exist(folder, base_out):
        print(f"\n⏭️  [{label}] 結果が既に存在するためスキップ: {os.path.join(base_out, folder)}")
        return
    print(f"\n{'='*60}")
    print(f"  シミュレーション開始: {label}  (mode={mode})")
    print(f"{'='*60}\n")
    cmd = [sys.executable, SIM_SCRIPT, mode]
    if route_file:
        cmd.extend(["--route-file", route_file, "--base-out", base_out])
    subprocess.run(cmd, check=True)
    print(f"\n✅ {label} 完了\n")


def parse_cli_args(argv):
    parser = argparse.ArgumentParser(
        description="実行したい制御手法を指定してシミュレーションし、比較レポートを生成します。",
    )
    parser.add_argument(
        "modes",
        nargs="*",
        help="実行する制御手法: none / gap / prediction / all（日本語名も可）",
    )
    parser.add_argument(
        "--route-dir",
        help=f"複数日分の .rou.xml が入ったフォルダ。未指定時の候補: {DEFAULT_ROUTE_DIR}",
    )
    parser.add_argument(
        "--route-file",
        action="append",
        default=[],
        help="実行する .rou.xml。複数指定する場合は --route-file を繰り返します。",
    )
    args = parser.parse_args(argv)

    explicit_all = any(mode.lower() == "all" for mode in args.modes)
    explicit_compare = any(mode.lower() in ("compare", "比較") for mode in args.modes)

    if explicit_all:
        selected = ALL_MODES
    elif explicit_compare:
        selected = REQUIRED_COMPARISON_MODES
    elif not args.modes:
        selected = None
    else:
        selected = []
        invalid = []
        seen = set()
        for raw_mode in args.modes:
            for token in raw_mode.split(","):
                token = token.strip()
                if not token:
                    continue
                mode = MODE_ALIASES.get(token.lower(), MODE_ALIASES.get(token))
                if mode == "compare":
                    for compare_mode, compare_label in REQUIRED_COMPARISON_MODES:
                        if compare_mode not in seen:
                            selected.append((compare_mode, compare_label))
                            seen.add(compare_mode)
                    continue
                if mode is None:
                    invalid.append(token)
                    continue
                if mode not in seen:
                    selected.append((mode, MODE_LABELS[mode]))
                    seen.add(mode)

        if invalid:
            valid = "none, gap, prediction, all（または 制御なし, ギャップ感応制御, 予測制御）"
            parser.error(f"未対応の制御手法です: {', '.join(invalid)} / 指定可能: {valid}")

    route_files = []
    if args.route_dir:
        route_dir = Path(args.route_dir)
        if not route_dir.exists() or not route_dir.is_dir():
            parser.error(f"--route-dir が見つからないかフォルダではありません: {route_dir}")
        route_files.extend(sorted(str(path) for path in route_dir.glob("*.rou.xml")))
        if not route_files:
            parser.error(f"--route-dir に .rou.xml が見つかりません: {route_dir}")

    for route_file in args.route_file:
        route_path = Path(route_file)
        if not route_path.exists() or not route_path.is_file():
            parser.error(f"--route-file が見つかりません: {route_path}")
        route_files.append(str(route_path))

    deduped_route_files = []
    seen_routes = set()
    for route_file in route_files:
        resolved = str(Path(route_file).resolve())
        if resolved not in seen_routes:
            deduped_route_files.append(resolved)
            seen_routes.add(resolved)

    if selected is None:
        selected = REQUIRED_COMPARISON_MODES if deduped_route_files else ALL_MODES

    if deduped_route_files:
        required_keys = {mode for mode, _ in REQUIRED_COMPARISON_MODES}
        selected_keys = {mode for mode, _ in selected}
        selected = _ordered_modes(selected_keys | required_keys)

    return selected, deduped_route_files


# ============================================================
# 比較レポート生成
# ============================================================
def read_summary(dirpath):
    path = os.path.join(dirpath, SUMMARY_FILE)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, encoding="utf-8-sig").fillna(0)
    result = {}
    for label in ["全体合計", "J交差点合計", "J以外合計"]:
        row = df[df["交差点"] == label]
        if row.empty:
            result[label] = (0, 0, 0)
        else:
            r = row.iloc[0]
            t1    = int(r.get("前半台数合計",    0) or 0)
            t2    = int(r.get("後半台数合計",    0) or 0)
            delay = int(r.get("遅れ時間合計(秒)", 0) or 0)
            result[label] = (t1, t2, delay)
    return result


def read_cycle(dirpath):
    path = os.path.join(dirpath, CYCLE_FILE)
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, encoding="utf-8-sig").fillna(0)
    return dict(zip(df["項目"].astype(str), df["値"]))


def generate_report(base_out=BASE_OUT, report_modes=None):
    print(f"\n{'='*60}")
    print("  比較レポート生成")
    print(f"{'='*60}\n")

    if report_modes is None:
        report_modes = ALL_MODES

    summary_data = {}
    cycle_data   = {}
    for mode, _ in report_modes:
        folder = MODE_FOLDERS[mode]
        dirpath = os.path.join(base_out, folder)
        sd = read_summary(dirpath)
        cd = read_cycle(dirpath)
        if sd is None:
            print(f"⚠️ [{folder}] 集計CSVが見つかりません: {dirpath}")
        summary_data[mode] = sd
        cycle_data[mode]   = cd

    columns = ["項目"]
    for mode, _ in report_modes:
        columns.append(MODE_REPORT_COLUMNS[mode])
        if mode != "none":
            columns.append(MODE_DIFF_COLUMNS.get(mode, f"差({MODE_REPORT_COLUMNS[mode]}vs制御なし)"))

    rows = []

    def add_blank():
        rows.append({c: "" for c in columns})

    def add_header(name):
        row = {c: "" for c in columns}
        row["項目"] = f"【{name}】"
        rows.append(row)

    def diff(v, base):
        return "" if (v == "" or base == "") else v - base

    def get_vals(mode, section_key):
        sd = summary_data.get(mode)
        if sd is None or section_key not in sd:
            return "", "", "", ""
        t1, t2, delay = sd[section_key]
        return t1, t2, t1 + t2, delay

    def cval(mode, key):
        v = cycle_data.get(mode, {}).get(key, "")
        if v == "":
            return ""
        try:
            return int(float(v))
        except Exception:
            return ""

    def add_metric_row(label, values_by_mode):
        base = values_by_mode.get("none", "")
        row = {"項目": label}
        for mode, _ in report_modes:
            value = values_by_mode.get(mode, "")
            row[MODE_REPORT_COLUMNS[mode]] = value
            if mode != "none":
                row[MODE_DIFF_COLUMNS.get(mode, f"差({MODE_REPORT_COLUMNS[mode]}vs制御なし)")] = diff(value, base)
        rows.append(row)

    # ── 待ち台数・遅れ時間 ────────────────────────────────────
    for section_name, section_key in SECTIONS:
        add_header(f"{section_name} 待ち台数・遅れ時間")
        values = {mode: get_vals(mode, section_key) for mode, _ in report_modes}
        for idx, label in enumerate(["前半台数", "後半台数", "合計台数", "遅れ時間(秒)"]):
            add_metric_row(label, {mode: vals[idx] for mode, vals in values.items()})
        add_blank()

    # ── サイクル数 ────────────────────────────────────────────
    add_header("サイクル数")
    for label, key in [
        ("Jサイクル総数",          "Jサイクル総数"),
        ("道路1制御サイクル数",     "道路1制御サイクル数"),
        ("道路1調整サイクル数",     "道路1調整サイクル数"),
        ("道路5制御サイクル数",     "道路5制御サイクル数"),
        ("道路5調整サイクル数",     "道路5調整サイクル数"),
        ("主道路GAP調整サイクル数", "主道路GAP調整サイクル数"),
        ("従道路GAP調整サイクル数", "従道路GAP調整サイクル数"),
    ]:
        add_metric_row(label, {mode: cval(mode, key) for mode, _ in report_modes})

    df_out = pd.DataFrame(rows, columns=columns)

    now_str  = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    out_name = f"制御比較_閾値{OFFSET_ADJUST_THRESHOLD}s_{now_str}.csv"
    os.makedirs(base_out, exist_ok=True)
    out_path = os.path.join(base_out, out_name)
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"✅ 比較レポート出力完了: {out_path}")
    print(df_out.to_string(index=False))


# ============================================================
# エントリーポイント
# ============================================================
if __name__ == "__main__":
    modes_to_run, route_files = parse_cli_args(sys.argv[1:])

    if route_files:
        run_contexts = [
            (route_file, os.path.join(BASE_OUT, "複数日", _route_run_name(route_file)))
            for route_file in route_files
        ]
    else:
        run_contexts = [(None, BASE_OUT)]

    for route_file, base_out in run_contexts:
        if route_file:
            print(f"\n🚗 車両データ: {route_file}")
            print(f"📁 出力ルート: {base_out}")
        print("▶ 実行対象: " + ", ".join(lbl for _, lbl in modes_to_run))

        for mode, lbl in modes_to_run:
            folder = MODE_FOLDERS[mode]
            run_simulation(
                mode,
                lbl,
                folder,
                base_out=base_out,
                route_file=route_file,
                skip_existing=bool(route_file),
            )
        generate_report(base_out=base_out, report_modes=modes_to_run)

    print("\n✅ 指定モード実行・比較レポート生成完了")
