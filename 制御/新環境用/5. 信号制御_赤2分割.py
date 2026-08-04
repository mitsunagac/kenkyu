import sys
import os
sys.stdout.reconfigure(encoding='utf-8')

os.environ['SUMO_HOME'] = r'C:\Program Files (x86)\Eclipse\Sumo'
sys.path.append(r'C:\Program Files (x86)\Eclipse\Sumo\tools')

import csv
import math
import re
import importlib.util
import xml.etree.ElementTree as ET
import torch
import traci
import sumolib
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
import warnings
import joblib

# ============================================================================
# 【赤2分割GNN 信号制御】 4. 信号制御_赤3分割.py の2分割版
# ----------------------------------------------------------------------------
# 3分割版との違いは次の3点だけ。他（サイクル長スケジュール、J交差点の青更新方式、
# 遅れ時間・待ち台数の全信号集計、ギャップ感応制御、時間距離図、各種ログ）は同一。
#   ① 予測モデル: 予測モデル/1. 赤時間2分割/道路{1,5}_分散評価あり
#                  [前半,後半] → 次サイクルの [前半,後半]（2入力2出力）
#   ② 計測      : 赤を2等分（RedSplit2Tracker）。3分割版と同じ初停車ビン方式。
#   ③ delta     : 旧2分割制御（1. 信号制御用_従道路も.py）の規則
#                  t3>t4 → +t3*2 / t3<t4 → −t4*2 / 同数 → 0（±10sクランプ）
#
# ★サイクル長は案A（最新プログラムどおり時間帯別 100〜130s）を採用。
#   2分割モデルの学習データ(12.22)は130s固定サイクルで計測されているため、赤の絶対長は
#   学習時と変わる。モデルは「赤を2等分した各区間の台数」を見るので比は保たれるが、
#   学習条件に厳密に合わせたい場合は 赤2分割_制御コア.py の CYCLE_SCHEDULE_BY_HOUR を
#   全時間帯130sにすること（比較対象の none/gap も同条件で回し直す必要がある）。
#
# 環境変数（3分割版の R3_* に対応する R2_*）:
#   R2_MODE=prediction|gap|pgap|pgapmin|npred|none   制御モード
#   R2_VARIANT=red2|none                             予測制御の有効/無効
#   R2_MODEL1 / R2_MODEL5                            使用するモデルフォルダ名
#   R2_EVENCANCEL=1                                  旧2分割の奇数制御/偶数打ち消しにする
#   R2_DEADBAND=2                                    前半後半の差がこの台数未満なら delta=0
#   R2_END / R2_GUI / R2_GAPSHORT                    終了時刻 / GUI / 予測ギャップ閾値
# ============================================================================

# ============================================================
# 赤2分割制御コアを読み込む
#   ・モデル(RedSplitPredictionGNN) / 予測(run_prediction_red2) / delta(delta_red2)
#   ・赤2分割の実時間計測(RedSplit2Tracker, 初停車ビン方式)
#   ・時間帯別サイクル長(get_target_cycle / build_scaled_logic) ※3分割コアと共用
# ============================================================
_core_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "赤2分割_制御コア.py")
_spec = importlib.util.spec_from_file_location("red2_core", _core_path)
red2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(red2)

# 到着間隔（W・N同時予測）コア: 予測ギャップ制御(pgap)で使用
_arr_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "到着間隔_制御コア.py")
_arr_spec = importlib.util.spec_from_file_location("arr_core", _arr_path)
arr = importlib.util.module_from_spec(_arr_spec)
_arr_spec.loader.exec_module(arr)

# ============================================================
# 制御案の選択（比較実験用フラグ）
#   "red2": 旧2分割制御の規則で delta を決める / "none": delta=0（予測はするが動かさない）
#   ※ CONTROL_MODE="prediction" のときのみ有効。CONTROL_MODE="none"/"gap" はこのフラグを無視。
#   3分割版のような複数案（1/2A/2B/3）は無い。2区間なので規則が1つしかないため。
# ============================================================
CONTROL_VARIANT = os.environ.get("R2_VARIANT", "red2")   # red2/none（環境変数 R2_VARIANT で上書き可）

# === delta の決め方 ===
#   "weighted"(既定) … ギャップ感応と同じく「初期状態 = BASE − 10s」から始め、
#                       予測台数に応じて足す:
#                         delta = -10 + (前半予測 × X1 + 後半予測 × X2) × 2
#                       重み付き台数 0台→BASE−10s、5台→BASE、10台→BASE+10s（上限）。
#   "compare"        … 旧2分割制御の規則（t3>t4 → +t3*2 / t3<t4 → −t4*2）。過去結果の再現用。
DELTA_RULE = os.environ.get("R2_RULE", "weighted")
DELTA_X1 = float(os.environ.get("R2_X1", str(red2.X1_DEFAULT)))   # 前半(t3)の重み
DELTA_X2 = float(os.environ.get("R2_X2", str(red2.X2_DEFAULT)))   # 後半(t4)の重み

# --- 交通量の多い時間帯だけ重みを上げる ---
# ピーク時は需要が大きく、通常の重み（x<1）では delta が上限に届かず主道路へ十分な青を
# 配分できない。そこでピーク時間帯に限り x1 を X1_MAX_PEAK(=1.5) まで上げられるようにする。
# 既定のピーク時間帯 = サイクル長が最長(130s)に設定されている時間帯（7-8時, 17-19時）。
#   R2_PEAK_HOURS="6-19"   のように時間帯を直接指定することもできる（複数は , 区切り）
#   R2_X1_PEAK / R2_X2_PEAK でピーク時の重みを指定（既定はピークも通常と同じ）
_max_cycle = max(c for _, _, c in red2.CYCLE_SCHEDULE_BY_HOUR)
_default_peak = ",".join(f"{a}-{b}" for a, b, c in red2.CYCLE_SCHEDULE_BY_HOUR if c == _max_cycle)
PEAK_HOURS_SPEC = os.environ.get("R2_PEAK_HOURS", _default_peak)
PEAK_RANGES = []
for _part in PEAK_HOURS_SPEC.split(","):
    _part = _part.strip()
    if not _part:
        continue
    _a, _b = _part.split("-")
    PEAK_RANGES.append((float(_a), float(_b)))

DELTA_X1_PEAK = float(os.environ.get("R2_X1_PEAK", str(DELTA_X1)))
DELTA_X2_PEAK = float(os.environ.get("R2_X2_PEAK", str(DELTA_X2)))
USE_PEAK_WEIGHTS = (DELTA_X1_PEAK != DELTA_X1) or (DELTA_X2_PEAK != DELTA_X2)

if DELTA_RULE == "weighted":
    red2.validate_weights(DELTA_X1, DELTA_X2)                      # 通常: 0 < x1 < 1
    red2.validate_weights(DELTA_X1_PEAK, DELTA_X2_PEAK, peak=True)  # ピーク: x1 は 1.5 まで可


def weights_at(step):
    """その時刻に使う (x1, x2) を返す。ピーク時間帯なら ピーク用の重み。"""
    if not USE_PEAK_WEIGHTS:
        return DELTA_X1, DELTA_X2
    h = (float(step) - PREP_TIME) / 3600.0
    for a, b in PEAK_RANGES:
        if a <= h < b:
            return DELTA_X1_PEAK, DELTA_X2_PEAK
    return DELTA_X1, DELTA_X2

# delta のデッドバンド(台): 旧ルール("compare")のみ有効。前半と後半の差がこの台数未満なら delta=0。
DELTA_DEADBAND = int(os.environ.get("R2_DEADBAND", "0"))

# 旧2分割制御の「奇数サイクル=予測で制御 / 偶数サイクル=前サイクルの打ち消し」を使うか。
# 既定 0（毎サイクル制御＋累積オフセット補正 = 3分割版と同じ方式）。
# ※ 2分割の delta は「赤の前半に多く停まる」傾向のため正（青延長）に偏りやすい。
#   学習データのテスト日で試算すると delta 平均は 道路1=+3.5s / 道路5=+3.0s（道路5は負が0回）。
#   旧プログラムが偶数サイクルで打ち消していたのは、この偏りを相殺するためと思われる。
#   既定の累積オフセット補正でもドリフトは抑制されるが、挙動を比べたい場合は R2_EVENCANCEL=1。
EVEN_CYCLE_CANCEL_PRED = os.environ.get("R2_EVENCANCEL", "0") == "1"

# 予測ギャップ(pgap)の緩めたギャップ閾値(秒)。R2_GAPSHORT で上書き可（既定2s）。
# 大きくするほど「本物でない一時的な隙間での早切り＝積み残し」を減らせる（安全マージン）。
PGAP_GAP_SHORT = int(os.environ.get("R2_GAPSHORT", "2"))

# 使用する予測モデルのフォルダ名（予測モデル/1. 赤時間2分割/ 配下）
MODEL_DIR_ROAD1 = os.environ.get("R2_MODEL1", "道路1_分散評価あり")
MODEL_DIR_ROAD5 = os.environ.get("R2_MODEL5", "道路5_分散評価あり")


shown_warnings = set()

def show_warning_once(message):
    if message not in shown_warnings:
        warnings.warn(message, FutureWarning)
        shown_warnings.add(message)

def get_time_label_and_step(step, total_steps_in_day=1330, num_classes=5, prep_time=1300):
    if step < prep_time:
        raise ValueError(f"❌ step={step} は準備時間 {prep_time} 秒未満です。時間ラベルは付与されません。")
    step_in_day = step % total_steps_in_day
    block_size = total_steps_in_day // num_classes
    time_label = step_in_day // block_size
    return time_label, step_in_day


# ============================================================
# 赤2分割の予測（コアモジュールに委譲）
#   wait_9x2: 全9道路の当該サイクル2分割 [9,2]（実台数）
#   戻り: 対象道路の次サイクル [t3,t4]（実台数）
# ============================================================
def run_prediction_red2(wait_9x2, target_node_index, model, scaler_wait, edge_index):
    return red2.run_prediction_red2(wait_9x2, target_node_index, model, scaler_wait, edge_index)


# ============================================================
# ログ出力クラス
# ============================================================
class TrafficLogger:
    def __init__(self, used_road_ids):
        self.used_road_ids = used_road_ids
        self.measurement_count = 0
        self.prediction_count = 0

    def log_measurement_data(self, road_id, t1_value, t2_minus_t1_value, step):
        self.measurement_count += 1

    def log_prediction_trigger(self, all_data, label=""):
        print(f"🔍 === 予測実行時の全道路データ {label}===")
        for road_id in self.used_road_ids:
            if all_data[road_id] is not None:
                t1, t2_minus_t1 = all_data[road_id]
                print(f"   道路{road_id}: t1={t1:2d}, t2-t1={t2_minus_t1:2d}")
            else:
                print(f"   道路{road_id}: データなし")
        print("=====================================")

    def log_prediction_result(self, count, t1_count, diff, pred_t3, pred_t4, step, label="道路1"):
        print(f"🔮 [{label}][{count:3d}回目] 入力: t1={t1_count:2d}, t2-t1={diff:2d} → 予測: t3={pred_t3:2d}, t4={pred_t4:2d} (step={step})")

    def log_data_incomplete(self, step, label=""):
        print(f"⏭️ {label}予測パス（step={step}）: データ未揃い → 破棄")


# ============================================================
# パス設定（出力先は CONTROL_MODE に応じて自動決定）
# ============================================================
sumocfg_path   = r"C:\Users\Tsukasa\Desktop\研究\予測\町モデルデータ\toyama_shouwa.sumocfg"
train_csv_path = r"C:\Users\Tsukasa\Desktop\研究\予測\★学習用データ\1. 赤時間2分割_学習データ_12.22\新環境待ち台数データ_12.22.csv"
adj_path       = r"C:\Users\Tsukasa\Desktop\研究\予測\新環境_隣接行列.csv"
BASE_OUT       = r"C:\Users\Tsukasa\Desktop\研究\制御\csv掃き出し"

# --- 道路1 モデル ---
model_path_road1  = r"C:\Users\Tsukasa\Desktop\研究\予測\予測モデル\0. 制御用_待ち台数単体(3.22)\道路1\GCN_epoch_best_model.pth"
scaler_path_road1 = r"C:\Users\Tsukasa\Desktop\研究\予測\予測モデル\0. 制御用_待ち台数単体(3.22)\道路1\scaler.pkl"

# --- 道路5 モデル ---
model_path_road5  = r"C:\Users\Tsukasa\Desktop\研究\予測\予測モデル\0. 制御用_待ち台数単体(3.22)\道路5\GCN_epoch_best_model.pth"
scaler_path_road5 = r"C:\Users\Tsukasa\Desktop\研究\予測\予測モデル\0. 制御用_待ち台数単体(3.22)\道路5\scaler.pkl"


# ============================================================
# 使用ノード・スケーラー・edge_index（道路1用・道路5用 共通）
# ============================================================
used_road_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]
used_road_ids     = [str(i + 1) for i in range(9)]   # 道路1～9

latest_data = {rid: None for rid in used_road_ids}

logger = TrafficLogger(used_road_ids)

# スケーラーは後で joblib.load するのでここでは構築しない
# （run_simulation 内で読み込む）

# edge_index 構築（隣接行列のみから。学習時と同一の構築方法）
adj_matrix        = pd.read_csv(adj_path, encoding='shift-jis', index_col=0).values
edge_index_raw    = np.array(np.nonzero(adj_matrix)).astype(np.int64)
mask              = np.isin(edge_index_raw[0], used_road_indices) & np.isin(edge_index_raw[1], used_road_indices)
edge_index_filtered = edge_index_raw[:, mask]
id_map            = {old: new for new, old in enumerate(used_road_indices)}
edge_index_mapped = np.vectorize(id_map.get)(edge_index_filtered)
edge_index        = torch.tensor(edge_index_mapped, dtype=torch.long)


# ============================================================
# 信号設定
# ============================================================
SIM_END_TIME = int(os.environ.get("R2_END", "86400"))   # 終了時刻(秒)。R2_END で上書き可（デバッグ短縮用）
PREP_TIME    = 1200
USE_GUI      = os.environ.get("R2_GUI", "0") == "1"      # R2_GUI=1 で sumo-gui、既定は headless sumo

# ============================================================
# 集計の時間窓（24hデータのうち、この時間帯だけに絞った集計も出力する）
#   step → 時刻(時) 変換: hour = (step - PREP_TIME) / 3600
#   6時〜19時のみで比較したい、という用途。全24hの集計も別に出す。
# ============================================================
ANALYSIS_START_HOUR = 6
ANALYSIS_END_HOUR   = 19


def step_to_hour(step):
    """シミュレーション step(秒) を「準備終了を0時とした時刻(時)」に変換。"""
    return (float(step) - PREP_TIME) / 3600.0


def in_analysis_window(step):
    """step が集計時間窓（既定 6〜19時）に入るか。"""
    h = step_to_hour(step)
    return ANALYSIS_START_HOUR <= h < ANALYSIS_END_HOUR

# ============================================================
# 制御モード設定
#   "prediction" : GNN予測ベース信号制御（道路1・道路5）
#   "gap"        : ギャップ感応制御（GapCycleController）
#   "none"       : 制御なし（固定サイクル・計測・遅れ時間ログのみ）
# ============================================================
CONTROL_MODE = os.environ.get("R2_MODE", "prediction")   # prediction/gap/none（環境変数 R2_MODE で上書き可）
# ※ 旧フラグ。現在はどの制御にも影響しない（未使用）。
#   ギャップ感応の累積オフセット補正は常時有効になり、予測制御側の
#   奇数制御/偶数打ち消しは R2_EVENCANCEL（EVEN_CYCLE_CANCEL_PRED）で切り替える。
ENABLE_EVEN_CYCLE_CANCEL = False

# ── オフセット調整パラメータ（ここを変更して閾値・補正量を調整）──────────────
# 累積制御偏差がこの秒数に達したとき調整サイクルを割り込ませる
OFFSET_ADJUST_THRESHOLD  = 20   # 閾値（秒）  例: 20 → ±20s 蓄積で調整開始
# 1調整サイクルあたりの補正量（秒）。最大増減時間と一致させること
OFFSET_ADJUST_PER_CYCLE  = 10   # 補正量（秒）例: 10s/サイクル → 20s 補正に 2サイクル必要
# ────────────────────────────────────────────────────────────────────────────

# 制御モード → 出力フォルダ名
_MODE_FOLDER_MAP = {
    "prediction": "予測制御",
    "gap":        "ギャップ感応制御",
    "pgap":       "予測ギャップ制御",
    "pgapmin":    "予測ギャップ_MIN調整",
    "npred":      "N単独制御",
    "none":       "制御なし",
}
# 出力は「csv掃き出し/赤2分割/<制御方式>/」にまとめる。
# 3分割版(4. 信号制御_赤3分割.py)は csv掃き出し 直下に出力するので、混ざらない。
_RESULT_ROOT = os.path.join(BASE_OUT, "赤2分割")
if CONTROL_MODE == "prediction":
    if CONTROL_VARIANT != "red2":
        _OUT_FOLDER = f"予測制御_{CONTROL_VARIANT}"
    elif DELTA_RULE == "weighted":
        # x1/x2 を変えて試すたびに別フォルダへ出す（前の結果を潰さない）
        _OUT_FOLDER = f"予測制御_x1={DELTA_X1:g}_x2={DELTA_X2:g}"
        if USE_PEAK_WEIGHTS:
            _OUT_FOLDER += f"_ピーク{PEAK_HOURS_SPEC}_x1={DELTA_X1_PEAK:g}_x2={DELTA_X2_PEAK:g}"
    else:
        _OUT_FOLDER = "予測制御_旧ルール"
    if EVEN_CYCLE_CANCEL_PRED:
        _OUT_FOLDER += "_偶数打消"
    if MODEL_DIR_ROAD1 != "道路1_分散評価あり" or MODEL_DIR_ROAD5 != "道路5_分散評価あり":
        _OUT_FOLDER += "_別モデル"
elif CONTROL_MODE == "pgap":
    _OUT_FOLDER = ("予測ギャップ制御" if PGAP_GAP_SHORT == 2
                   else f"予測ギャップ制御_gap{PGAP_GAP_SHORT}s")
else:
    _OUT_FOLDER = _MODE_FOLDER_MAP[CONTROL_MODE]
OUT_DIR      = os.path.join(_RESULT_ROOT, _OUT_FOLDER)
log_dir_path = os.path.join(OUT_DIR, "各道路計測ログ")
_delay_dir   = os.path.join(OUT_DIR, "遅れ時間")
os.makedirs(_delay_dir,   exist_ok=True)
os.makedirs(log_dir_path, exist_ok=True)
print(f"🚦 赤2分割GNN制御  mode={CONTROL_MODE} / variant={CONTROL_VARIANT}")
if CONTROL_MODE == "prediction":
    if DELTA_RULE == "weighted":
        print(f"   delta規則: 需要重み付け  delta = -10 + (前半×{DELTA_X1:g} + 後半×{DELTA_X2:g})×2"
              f"  （重み付き 0台→BASE-10s / 5台→BASE / 10台→BASE+10s）")
        if USE_PEAK_WEIGHTS:
            print(f"   ピーク時間帯({PEAK_HOURS_SPEC}時)は 前半×{DELTA_X1_PEAK:g} + 後半×{DELTA_X2_PEAK:g}")
    else:
        print(f"   delta規則: 旧2分割（t3>t4→+t3*2 / t3<t4→-t4*2） デッドバンド={DELTA_DEADBAND}台")
    print(f"   delta方式: {'奇数制御/偶数打ち消し（旧2分割方式）' if EVEN_CYCLE_CANCEL_PRED else '毎サイクル制御＋累積補正'}")
    print(f"   使用モデル: 道路1={MODEL_DIR_ROAD1} / 道路5={MODEL_DIR_ROAD5}")
print(f"📁 出力フォルダ: {OUT_DIR}")

TRAFFIC_LIGHT_CONFIG = {
    "A": [
    {"id": "11", "edges": ["-E43", "-E44", "-973892289#0"], "red": "rrryyyyrrryyyyy", "green": "rrrrrrrrrrrrrrr"},
    {"id": "12", "edges": ["E45"], "red": "yyyyrrrrryyyyrrrrr", "green": "rrrrrrrrrrrrrrr"},
    {"id": "13", "edges": ["E46"], "red": "yyyyrrrrryyyyrrrrr", "green": "rrrrrrrrrrrrrrr"},
    {"id": "14", "edges": ["-E48"], "red": "rrryyyyrrryyyyy", "green": "rrrrrrrrrrrrrrrr"}
    ],
    "B": [
    {"id": "15", "edges": ["-155398386#0"], "red": "rrryyyrrryyy", "green": "rrrrrrrrrrrr"},
    {"id": "16", "edges": ["-973892417#0"], "red": "yyyrrryyyrrr", "green": "rrrrrrrrrrrr"},
    {"id": "17", "edges": ["622541624#11"], "red": "yyyrrryyyrrr", "green": "rrrrrrrrrrrr"},
    {"id": "18", "edges": ["155389111"], "red": "rrryyyrrryyy", "green": "rrrrrrrrrrrr"}
    ],
    "C": [
    {"id": "19", "edges": ["E128"], "red": "rrrryyyrrryyy", "green": "rrrrrrrrrrrrr"},
    {"id": "20", "edges": ["-E130"], "red": "yyyyrrryyyrrr", "green": "rrrrrrrrrrrrr"},
    {"id": "21", "edges": ["-E129"], "red": "yyyyrrryyyrrr", "green": "rrrrrrrrrrrrr"},
    {"id": "22", "edges": ["-E131"], "red": "rrrryyyrrryyy", "green": "rrrrrrrrrrrrr"}
    ],
    "D": [
    {"id": "23", "edges": ["-543210000#1", "-E41"], "red": "rrryyyyrrryyyy", "green": "rrrrrrrrrrrrrr"},
    {"id": "24", "edges": ["155394941#2"], "red": "yyyrrrryyyrrrr", "green": "rrrrrrrrrrrrrr"},
    {"id": "25", "edges": ["155398386#1"], "red": "yyyrrrryyyrrrr", "green": "rrrrrrrrrrrrrr"},
    {"id": "26", "edges": ["973892290", "E44"], "red": "rrryyyyrrryyyy", "green": "rrrrrrrrrrrrrr"},
    ],
    "E": [
    {"id": "2", "edges": ["E40", "-E22"], "red": "rrryyyyrrryyyy", "green": "rrrrrrrrrrrrrr"},
    {"id": "27", "edges": ["-E27"], "red": "yyyrrrryyyrrrr", "green": "rrrrrrrrrrrrrr"},
    {"id": "28", "edges": ["E42", "E41"], "red": "rrryyyyrrryyyy", "green": "rrrrrrrrrrrrrr"},
    ],
    "F": [
    {"id": "3", "edges":  ["E3", "E1", "-E34", "E35"], "red": "rrryyyyrrryyyy", "green": "rrrrrrrrrrrrrr"},
    {"id": "29", "edges": ["155396812#0"], "red": "yyyrrrryyyrrrr", "green": "rrrrrrrrrrrrrr"},
    {"id": "30", "edges": ["-155397698#1"], "red": "yyyrrrryyyrrrr", "green": "rrrrrrrrrrrrrr"},
    {"id": "31", "edges": ["E26", "E22"], "red": "rrryyyyrrryyyy", "green": "rrrrrrrrrrrrrr"},
    ],
    "G": [
    {"id": "32", "edges": ["E62"], "red": "rrrryyyrrrryyy", "green": "rrrrrrrrrrrrrr"},
    {"id": "33", "edges": ["-E58", "-E59"], "red": "yyyyrrryyyyrrr", "green": "rrrrrrrrrrrrrr"},
    {"id": "34", "edges": ["E61", "E60"], "red": "yyyyrrryyyyrrr", "green": "rrrrrrrrrrrrrr"},
    {"id": "35", "edges": ["-E63"], "red": "rrrryyyrrrryyy", "green": "rrrrrrrrrrrrrr"},
    ],
    "H": [
    {"id": "36", "edges": ["155391024"], "red": "rrrryyyrrrryyy", "green": "rrrrrrrrrrrrrr"},
    {"id": "37", "edges": ["-E56"], "red": "yyyyrrryyyyrrr", "green": "rrrrrrrrrrrrrr"},
    {"id": "38", "edges": ["-155398822#23", "E59"], "red": "yyyyrrryyyyrrr", "green": "rrrrrrrrrrrrrr"},
    {"id": "39", "edges": ["155390837#1"], "red": "rrrryyyrrrryyy", "green": "rrrrrrrrrrrrrr"},
    ],
    "I": [
    {"id": "40", "edges": ["-155387788"], "red": "rrrryyyrrrryyy", "green": "rrrrrrrrrrrrrr"},
    {"id": "41", "edges": ["155398822#21"], "red": "yyyyrrryyyyrrr", "green": "rrrrrrrrrrrrrr"},
    {"id": "4",  "edges": ["E57"], "red": "yyyyrrryyyyrrr", "green": "rrrrrrrrrrrrrr"},
    {"id": "42", "edges": ["155387430"], "red": "rrrryyyrrrryyy", "green": "rrrrrrrrrrrrrr"},
    ],
    "J": [
    {"id": "1",  "edges": ["E24", "-E23", "-E33"], "red": "rrrryyyyrrrryyyy", "green": "rrrrrrrrrrrrrrrr"},
    {"id": "5",  "edges": ["E37", "E36"],           "red": "yyyyrrrryyyyrrrr", "green": "rrrrrrrrrrrrrrrr"},
    {"id": "6",  "edges": ["-E25", "-E69"],         "red": "yyyyrrrryyyyrrrr", "green": "rrrrrrrrrrrrrrrr"},
    {"id": "10", "edges": ["-E35", "E34", "-E1"],   "red": "rrrryyyyrrrryyyy", "green": "rrrrrrrrrrrrrrrr"}
    ],
    "K": [
    {"id": "43", "edges": ["E71"], "red": "rrrrrryy", "green": "rrrrrrrr"},
    {"id": "7",  "edges": ["155398822#14", "155398822#13"], "red": "yyyyyyrr", "green": "rrrrrrrr"},
    {"id": "44", "edges": ["E70"], "red": "yyyyyyrr", "green": "rrrrrrrr"}
    ],
    "L": [
    {"id": "45", "edges": ["E76"], "red": "rrrryyyrrrryyy", "green": "rrrrrrrrrrrrrr"},
    {"id": "46", "edges": ["-E74", "-E75"], "red": "yyyyrrryyyyrrr", "green": "rrrrrrrrrrrrrr"},
    {"id": "47", "edges": ["E73", "E72"], "red": "yyyyrrryyyyrrr", "green": "rrrrrrrrrrrrrr"},
    {"id": "48", "edges": ["E77"], "red": "rrrryyyrrrryyy", "green": "rrrrrrrrrrrrrr"}
    ],
    "M": [
    {"id": "8",  "edges": ["E31", "E30"], "red": "rrryyyyrrryyyy", "green": "rrrrrrrrrrrrrr"},
    {"id": "49", "edges": ["155390062#6"], "red": "yyyrrrryyyrrrr", "green": "rrrrrrrrrrrrrr"},
    {"id": "50", "edges": ["155389284#1"], "red": "yyyrrrryyyrrrr", "green": "rrrrrrrrrrrrrr"},
    {"id": "51", "edges": ["543210000#15", "-E32"], "red": "rrryyyyrrryyyy", "green": "rrrrrrrrrrrrrr"}
    ],
    "N": [
    {"id": "9",  "edges": ["E20", "E19"], "red": "rrryyyyrrryyyy", "green": "rrrrrrrrrrrrrr"},
    {"id": "52", "edges": ["155390603#0"], "red": "yyyrrrryyyrrrr", "green": "rrrrrrrrrrrrrr"},
    {"id": "53", "edges": ["-155385264#1"], "red": "yyyrrrryyyrrrr", "green": "rrrrrrrrrrrrrr"},
    {"id": "54", "edges": ["-E21"], "red": "rrryyyyrrryyyy", "green": "rrrrrrrrrrrrrr"}
    ],
    "O": [
    {"id": "55", "edges": ["E18"], "red": "rrryyyyrrryyyy", "green": "rrrrrrrrrrrrrr"},
    {"id": "56", "edges": ["-155389447#1"], "red": "yyyrrrryyyrrrr", "green": "rrrrrrrrrrrrrr"},
    {"id": "57", "edges": ["155387086#0"], "red": "yyyrrrryyyrrrr", "green": "rrrrrrrrrrrrrr"},
    {"id": "58", "edges": ["-E17", "E16"], "red": "rrryyyyrrryyyy", "green": "rrrrrrrrrrrrrr"}
    ],
    "P": [
    {"id": "59", "edges": ["E13", "E12"], "red": "rrryyyyrrryyyy", "green": "rrrrrrrrrrrrrr"},
    {"id": "60", "edges": ["-E14"], "red": "yyyrrrryyyrrrr", "green": "rrrrrrrrrrrrrr"},
    {"id": "61", "edges": ["-E15"], "red": "yyyrrrryyyrrrr", "green": "rrrrrrrrrrrrrr"},
    {"id": "62", "edges": ["-E18"], "red": "rrryyyyrrryyyy", "green": "rrrrrrrrrrrrrr"}
    ],
}

prediction_count = 0

# ============================================================
# A〜P 全信号の 遅れ時間 & 待ち台数 計測（表4/表5 用）
#   ・edge_to_signal: 監視エッジ → 信号(A〜P) の対応
#   ・遅れ時間 = 毎ステップ停車中(speed≤0)の車を計上（1ステップ=1秒）＝停車車両×秒
#   ・待ち台数 = 延べ停車台数（車が新規に停車した瞬間だけ1計上。停車継続中は重複しない）
#   ・全24h と 6〜19時窓 の両方を積算。対象交差点=J / 他交差点=J以外 / 街全体=全信号。
# ============================================================
TARGET_SIGNAL = "J"   # 対象交差点（表5の「対象交差点」）

edge_to_signal = {}
for _sig, _configs in TRAFFIC_LIGHT_CONFIG.items():
    for _conf in _configs:
        for _e in _conf["edges"]:
            edge_to_signal[_e] = _sig

signal_delay_full = defaultdict(int)   # 信号 → 停車車両×秒（全24h）
signal_delay_win  = defaultdict(int)   # 信号 → 停車車両×秒（6〜19時窓）
signal_queue_full = defaultdict(int)   # 信号 → 延べ停車台数（全24h）
signal_queue_win  = defaultdict(int)   # 信号 → 延べ停車台数（6〜19時窓）
_EMPTY_SET = frozenset()               # 待ち台数の新規停車判定用（前ステップ既定値）

# ── 時間帯別（signal_delay方式）集計: hour(0〜23) → 値 ──────────────────────
#   遅れ = 停車車両×秒（毎ステップ1回）／待ち = 延べ新規停車台数（signal方式と同基準）
#   J = 対象交差点 / all = 街全体（全信号）。時=step_to_hour（準備終了=0時）。
delay_hour_J   = defaultdict(int)      # 時 → J交差点の遅れ（秒）
delay_hour_all = defaultdict(int)      # 時 → 街全体の遅れ（秒）
queue_hour_J   = defaultdict(int)      # 時 → J交差点の待ち台数
queue_hour_all = defaultdict(int)      # 時 → 街全体の待ち台数
# ── 時間帯別: 青時間・Jサイクル数 ──────────────────────────────────────────
green_hour_sum = {"主道路": defaultdict(float), "従道路": defaultdict(float)}  # 時→青秒合計
green_hour_cnt = {"主道路": defaultdict(int),   "従道路": defaultdict(int)}    # 時→緑回数
cycle_hour_cnt = defaultdict(int)      # 時 → Jサイクル数


def write_signal_delay_summary():
    """A〜P 各信号の 待ち台数・遅れ時間 を、全24h と 6〜19時窓 で信号別に出力し、
    対象交差点(J)/他交差点(J以外)/街全体 の合計も付ける（表4・表5 を作りやすい形）。"""
    out_path = os.path.join(OUT_DIR,
                            f"信号別_待ち台数_遅れ_全24h_と_{ANALYSIS_START_HOUR}-{ANALYSIS_END_HOUR}時.csv")
    win = f"{ANALYSIS_START_HOUR}-{ANALYSIS_END_HOUR}時"
    sigs = sorted(set(edge_to_signal.values()))
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["区分", f"待ち台数_全24h", f"待ち台数_{win}",
                    f"遅れ時間_全24h(秒)", f"遅れ時間_{win}(秒)"])
        # 信号別
        for s in sigs:
            w.writerow([f"信号{s}", signal_queue_full.get(s, 0), signal_queue_win.get(s, 0),
                        signal_delay_full.get(s, 0), signal_delay_win.get(s, 0)])
        # 集計（対象交差点J / 他交差点 / 街全体）
        j_qf = signal_queue_full.get(TARGET_SIGNAL, 0); j_qw = signal_queue_win.get(TARGET_SIGNAL, 0)
        j_df = signal_delay_full.get(TARGET_SIGNAL, 0); j_dw = signal_delay_win.get(TARGET_SIGNAL, 0)
        all_qf = sum(signal_queue_full.values()); all_qw = sum(signal_queue_win.values())
        all_df = sum(signal_delay_full.values()); all_dw = sum(signal_delay_win.values())
        w.writerow([])
        w.writerow(["対象交差点(J)",   j_qf,            j_qw,            j_df,            j_dw])
        w.writerow(["他交差点(J以外)", all_qf - j_qf,   all_qw - j_qw,   all_df - j_df,   all_dw - j_dw])
        w.writerow(["街全体",          all_qf,          all_qw,          all_df,          all_dw])
    print(f"✅ 信号別 待ち台数・遅れ 集計を出力: {out_path}")
    print(f"   [街全体] 待ち: 全24h={all_qf} / {win}={all_qw}  遅れ: 全24h={all_df}s / {win}={all_dw}s")
    print(f"   [対象J ] 待ち: 全24h={j_qf} / {win}={j_qw}  遅れ: 全24h={j_df}s / {win}={j_dw}s")


def write_hourly_summary():
    """時間帯別（0〜23時）に、signal_delay方式の 遅れ・待ち台数（J/街全体）と、
    Jサイクル数・平均青時間（主道路=道路1/従道路=道路5）を出力する。
    どの時間帯で青が長い/短いか、遅れがどこに出るかを見る用。"""
    out_path = os.path.join(OUT_DIR, "時間帯別_signal方式_遅れ待ち_青サイクル.csv")

    def _avg(sum_d, cnt_d, h):
        c = cnt_d.get(h, 0)
        return round(sum_d.get(h, 0) / c, 1) if c else ""

    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["時", "Jサイクル数",
                    "主道路(道1)平均青(s)", "主道路 緑回数",
                    "従道路(道5)平均青(s)", "従道路 緑回数",
                    "J遅れ(秒)", "J待ち台数", "街全体遅れ(秒)", "街全体待ち台数"])
        tot = {"cyc": 0, "gm_s": 0.0, "gm_c": 0, "gs_s": 0.0, "gs_c": 0,
               "jd": 0, "jq": 0, "ad": 0, "aq": 0}
        for h in range(24):
            w.writerow([f"{h:02d}時", cycle_hour_cnt.get(h, 0),
                        _avg(green_hour_sum["主道路"], green_hour_cnt["主道路"], h),
                        green_hour_cnt["主道路"].get(h, 0),
                        _avg(green_hour_sum["従道路"], green_hour_cnt["従道路"], h),
                        green_hour_cnt["従道路"].get(h, 0),
                        delay_hour_J.get(h, 0), queue_hour_J.get(h, 0),
                        delay_hour_all.get(h, 0), queue_hour_all.get(h, 0)])
            tot["cyc"]  += cycle_hour_cnt.get(h, 0)
            tot["gm_s"] += green_hour_sum["主道路"].get(h, 0); tot["gm_c"] += green_hour_cnt["主道路"].get(h, 0)
            tot["gs_s"] += green_hour_sum["従道路"].get(h, 0); tot["gs_c"] += green_hour_cnt["従道路"].get(h, 0)
            tot["jd"]   += delay_hour_J.get(h, 0);   tot["jq"] += queue_hour_J.get(h, 0)
            tot["ad"]   += delay_hour_all.get(h, 0); tot["aq"] += queue_hour_all.get(h, 0)
        w.writerow([])
        w.writerow(["合計/平均", tot["cyc"],
                    round(tot["gm_s"] / tot["gm_c"], 1) if tot["gm_c"] else "", tot["gm_c"],
                    round(tot["gs_s"] / tot["gs_c"], 1) if tot["gs_c"] else "", tot["gs_c"],
                    tot["jd"], tot["jq"], tot["ad"], tot["aq"]])
    print(f"✅ 時間帯別（signal方式・青/サイクル）集計を出力: {out_path}")
    # どの時間が青最長/最短か（主道路）をコンソールにも
    _mavg = {h: green_hour_sum["主道路"][h] / green_hour_cnt["主道路"][h]
             for h in green_hour_cnt["主道路"] if green_hour_cnt["主道路"][h]}
    if _mavg:
        _hmax = max(_mavg, key=_mavg.get); _hmin = min(_mavg, key=_mavg.get)
        print(f"   [主道路青] 最長={_hmax:02d}時 {_mavg[_hmax]:.1f}s / 最短={_hmin:02d}時 {_mavg[_hmin]:.1f}s")

# ============================================================
# ログファイル設定
# ============================================================

# === 道路1 予測結果ログ ===
prediction_log_path = os.path.join(OUT_DIR, "予測結果_道路1.csv")
prediction_log_file = open(prediction_log_path, "w", newline="", encoding="utf-8-sig")
prediction_writer   = csv.writer(prediction_log_file)
prediction_writer.writerow(["回数", "step", "予測t3", "予測t4", "予測合計"])

# === 道路5 予測結果ログ ===
prediction_log_path_road5 = os.path.join(OUT_DIR, "予測結果_道路5.csv")
prediction_log_file_road5 = open(prediction_log_path_road5, "w", newline="", encoding="utf-8-sig")
prediction_writer_road5   = csv.writer(prediction_log_file_road5)
prediction_writer_road5.writerow(["回数", "step", "予測t3", "予測t4", "予測合計"])

# === フェーズ時間ログ ===
phase_log_path = os.path.join(OUT_DIR, "信号A_フェーズ時間ログ.csv")
phase_log_file = open(phase_log_path, "w", newline="", encoding="utf-8-sig")
phase_writer   = csv.writer(phase_log_file)
phase_writer.writerow(["回数", "step", "パターンA(秒)", "パターンB(秒)", "予測t3", "予測t4"])

# === 道路5 フェーズ時間ログ ===
phase_log_path_road5 = os.path.join(OUT_DIR, "信号J_フェーズ時間ログ_道路5.csv")
phase_log_file_road5 = open(phase_log_path_road5, "w", newline="", encoding="utf-8-sig")
phase_writer_road5   = csv.writer(phase_log_file_road5)
phase_writer_road5.writerow(["回数", "step", "パターンA(秒)", "パターンB(秒)", "予測t3", "予測t4"])

# === 無駄青ログ ===
wasted_green_log_path = os.path.join(OUT_DIR, "無駄青時間ログ.csv")
wasted_green_file     = open(wasted_green_log_path, "w", newline="", encoding="utf-8-sig")
wasted_green_writer   = csv.writer(wasted_green_file)
wasted_green_writer.writerow(["回数", "step", "監視秒数", "無駄青時間(s)"])

# === 遅れ時間ログ（遅れ時間サブフォルダ内）===
delay_log_path = os.path.join(_delay_dir, "信号機A_遅れ時間ログ_主道路(南).csv")
delay_log_file = open(delay_log_path, "w", newline="", encoding="utf-8-sig")
delay_writer   = csv.writer(delay_log_file)
delay_writer.writerow(["回数", "step", "前半台数", "後半台数", "delta", "okure", "遅れ時間(秒)"])

noth_delay_log_path = os.path.join(_delay_dir, "信号機A_遅れ時間ログ_主道路(北).csv")
noth_delay_log_file = open(noth_delay_log_path, "w", newline="", encoding="utf-8-sig")
noth_delay_writer   = csv.writer(noth_delay_log_file)
noth_delay_writer.writerow(["回数", "step", "前半台数", "後半台数", "delta", "okure", "遅れ時間(秒)"])

delay_minor_log_path = os.path.join(_delay_dir, "信号機A_遅れ時間_従道路.csv")
delay_minor_log_file = open(delay_minor_log_path, "w", newline="", encoding="utf-8-sig")
delay_minor_writer   = csv.writer(delay_minor_log_file)
delay_minor_writer.writerow(["回数", "step", "道路ID", "前半台数", "後半台数", "delta", "okure", "遅れ時間(秒)"])

delay_log_path_road5 = os.path.join(_delay_dir, "信号機J_遅れ時間ログ_道路5.csv")
delay_log_file_road5 = open(delay_log_path_road5, "w", newline="", encoding="utf-8-sig")
delay_writer_road5   = csv.writer(delay_log_file_road5)
delay_writer_road5.writerow(["回数", "step", "前半台数", "後半台数", "delta", "okure", "遅れ時間(秒)"])

# === 全道路遅れ時間ログ ===
all_delay_log_path = os.path.join(_delay_dir, "全遅れ時間ログ_主道路.csv")
all_delay_log_file = open(all_delay_log_path, "w", newline="", encoding="utf-8-sig")
all_delay_writer   = csv.writer(all_delay_log_file)
all_delay_writer.writerow(["回数", "step", "道路ID", "前半台数", "後半台数", "okure", "遅れ時間(秒)"])

all_delay_minor_log_path = os.path.join(_delay_dir, "全遅れ時間_従道路.csv")
all_delay_minor_log_file = open(all_delay_minor_log_path, "w", newline="", encoding="utf-8-sig")
all_delay_minor_writer   = csv.writer(all_delay_minor_log_file)
all_delay_minor_writer.writerow(["回数", "step", "道路ID", "前半台数", "後半台数", "okure", "遅れ時間(秒)"])

green_log_path = os.path.join(OUT_DIR, "green_durations.csv")
green_log_file = open(green_log_path, "w", newline="", encoding="utf-8-sig")
green_writer   = csv.writer(green_log_file)
green_writer.writerow(["回数", "step", "種類", "青時間(秒)"])

queue_delay_log_path = os.path.join(OUT_DIR, "queue_and_delay.csv")
queue_delay_file     = open(queue_delay_log_path, "w", newline="", encoding="utf-8-sig")
queue_delay_writer   = csv.writer(queue_delay_file)
queue_delay_writer.writerow(["回数", "step", "道路ID", "待ち台数", "合計遅れ時間(秒)"])


# ============================================================
# 初期フェーズ時間（固定）
# ============================================================
# ============================================================
# J交差点の制御適用（時間帯スケジュール青 + GNN delta + 累積補正）
# ------------------------------------------------------------
# ・時間帯別サイクル長でスケールした青(phase0=道路1, phase5=道路5)を基準に、
#   予測から求めた delta を各サイクル加算する（毎サイクル「スケジュール青+delta」で
#   compound しない）。
# ・累積が OFFSET_ADJUST_THRESHOLD に達したら調整サイクルで押し戻す（サイクル長ドリフト補正）。
# ・道路1=phase0、道路5=phase5（現ネットのJ位相。旧 phase6 は旧ネットの名残なので不使用）。
# ============================================================
J_GREEN_PHASES = (0, 5)                 # 制御対象の青フェーズ（道路1, 道路5）
J_BASE_GREEN   = {0: 63.0, 5: 29.0}     # 120sネットでのJ基準青（build_scaled_logicのbase値）

# J制御の状態（青フェーズごと）
j_scheduled_green = dict(J_BASE_GREEN)   # 現在の目標サイクルでのスケジュール青
j_applied_delta   = {0: 0, 5: 0}
j_cumulative      = {0: 0, 5: 0}
j_adjust_pending  = {0: False, 5: False}


# ── 可動範囲の上限 ─────────────────────────────────────────────────────────
# 目標サイクル長が変わっても、ずれの上限は常にこの2つで決まる（絶対秒数で固定）。
#   ・J_GREEN_DEV_CLAMP … 各青フェーズが「その時間帯のスケジュール青」からずれてよい幅
#   ・J_CYCLE_DEV_CLAMP … J のサイクル長が「その時間帯の目標サイクル長」からずれてよい幅
#     phase0 と phase5 は独立に制御されるため、両方が同じ向きに振れると
#     サイクル長のずれは単純合計になる。そこで delta を決める際、もう一方の
#     フェーズが既に使っている分を差し引いて枠内に収める。
# ※ 予測制御・ギャップ感応制御の両方に同じ枠を適用する。
J_GREEN_DEV_CLAMP = 10
J_CYCLE_DEV_CLAMP = int(os.environ.get("R2_CYCLECAP", "10"))
J_MIN_GREEN       = 5.0    # 青の絶対下限（これを下回る短縮はしない）


def update_j_scheduled_green(target_cycle):
    """目標サイクル長に応じてJの青(phase0/phase5)のスケジュール値を更新。
    build_scaled_logic と同じ配分（青2本に (target-120)/2 ずつ）。"""
    per = (target_cycle - red2.BASE_CYCLE_LENGTH) / 2.0
    for p in J_GREEN_PHASES:
        j_scheduled_green[p] = J_BASE_GREEN[p] + per


def clamp_delta_for_cycle(phase_idx, delta, other_delta=None):
    """delta を「自フェーズ ±J_GREEN_DEV_CLAMP」かつ
    「もう一方のフェーズとの合計が ±J_CYCLE_DEV_CLAMP 以内」に収める。
    サイクル長が 100s でも 130s でも、この幅は絶対秒数で変わらない。"""
    other = J_GREEN_PHASES[1] if phase_idx == J_GREEN_PHASES[0] else J_GREEN_PHASES[0]
    od = j_applied_delta[other] if other_delta is None else other_delta
    lo = max(-J_GREEN_DEV_CLAMP, -J_CYCLE_DEV_CLAMP - od)
    hi = min( J_GREEN_DEV_CLAMP,  J_CYCLE_DEV_CLAMP - od)
    if lo > hi:          # 相手が枠を使い切っている場合は動かさない
        return 0
    return int(max(lo, min(delta, hi)))


def green_bounds_for_phase(phase_idx, other_delta=None):
    """そのフェーズが取ってよい青時間の [下限, 上限]（秒）を返す。
    スケジュール青 ± J_GREEN_DEV_CLAMP を基本に、サイクル長の枠
    （もう一方のフェーズのずれとの合計 ±J_CYCLE_DEV_CLAMP）で更に絞る。
    ギャップ感応制御はこの範囲内で青を切る。"""
    sched = j_scheduled_green[phase_idx]
    lo_d = clamp_delta_for_cycle(phase_idx, -J_GREEN_DEV_CLAMP, other_delta)
    hi_d = clamp_delta_for_cycle(phase_idx,  J_GREEN_DEV_CLAMP, other_delta)
    return max(J_MIN_GREEN, sched + lo_d), max(J_MIN_GREEN, sched + hi_d)


def apply_j_logic():
    """Jの現在ロジックの phase0/phase5 を (スケジュール青 + applied_delta) に更新。他フェーズは保持。"""
    from traci._trafficlight import Phase
    logic  = traci.trafficlight.getAllProgramLogics("J")[0]
    phases = logic.getPhases()
    new_phases = []
    for i, ph in enumerate(phases):
        if i in J_GREEN_PHASES:
            dur = max(5.0, j_scheduled_green[i] + j_applied_delta[i])
        else:
            dur = ph.duration
        new_phases.append(Phase(duration=dur, state=ph.state,
                                minDur=getattr(ph, "minDur", dur),
                                maxDur=getattr(ph, "maxDur", dur)))
    logic.phases = new_phases
    traci.trafficlight.setProgramLogic("J", logic)


# 奇数/偶数サイクル方式（R2_EVENCANCEL=1）用の状態: 青フェーズごとのサイクル数と前回delta
j_cycle_count   = {0: 0, 5: 0}
j_last_delta    = {0: 0, 5: 0}


def control_j_road(phase_idx, road_id, t34, step, count):
    """予測 [t3,t4]（次サイクルの赤 前半/後半）から delta を求め、phase_idx の青を更新。
    戻り: 適用delta。CONTROL_MODE!='prediction' または CONTROL_VARIANT=='none' なら制御しない。

    2つの方式を切り替えられる:
      ・既定（EVEN_CYCLE_CANCEL_PRED=False）… 毎サイクル制御し、累積が
        OFFSET_ADJUST_THRESHOLD に達したら調整サイクルで押し戻す（3分割版と同じ）。
      ・R2_EVENCANCEL=1 … 旧2分割制御と同じく、奇数サイクルで予測制御・偶数サイクルで
        前サイクルの delta を打ち消す（累積は常に0付近に戻る）。
    """
    if CONTROL_MODE != "prediction" or CONTROL_VARIANT == "none":
        j_applied_delta[phase_idx] = 0
        apply_j_logic()
        return 0

    if EVEN_CYCLE_CANCEL_PRED:
        # ===== 旧2分割方式: 奇数=制御 / 偶数=打ち消し =====
        j_cycle_count[phase_idx] += 1
        if j_cycle_count[phase_idx] % 2 == 1:
            _x1, _x2 = weights_at(step)
            delta = (red2.delta_red2_weighted(t34, _x1, _x2)
                     if DELTA_RULE == "weighted"
                     else red2.delta_red2(t34, deadband=DELTA_DEADBAND))
            j_last_delta[phase_idx] = delta
            _tag = f"奇数#{j_cycle_count[phase_idx]}"
        else:
            delta = -j_last_delta[phase_idx]
            _tag = f"偶数#{j_cycle_count[phase_idx]}(打ち消し)"
        j_cumulative[phase_idx] += delta
    elif j_adjust_pending[phase_idx]:
        # 調整サイクル：累積を OFFSET_ADJUST_PER_CYCLE で押し戻す
        delta = -OFFSET_ADJUST_PER_CYCLE if j_cumulative[phase_idx] > 0 else OFFSET_ADJUST_PER_CYCLE
        j_adjust_pending[phase_idx] = False
        j_cumulative[phase_idx] += delta
        if abs(j_cumulative[phase_idx]) >= OFFSET_ADJUST_PER_CYCLE:
            j_adjust_pending[phase_idx] = True
        _tag = "調整"
    else:
        _x1, _x2 = weights_at(step)
        delta = (red2.delta_red2_weighted(t34, _x1, _x2)
                 if DELTA_RULE == "weighted"
                 else red2.delta_red2(t34, deadband=DELTA_DEADBAND))
        j_cumulative[phase_idx] += delta
        if abs(j_cumulative[phase_idx]) >= OFFSET_ADJUST_THRESHOLD:
            j_adjust_pending[phase_idx] = True
        _tag = "通常"

    # === 可動範囲の適用 ===
    # 自フェーズ ±J_GREEN_DEV_CLAMP に加え、もう一方のフェーズとの合計が
    # ±J_CYCLE_DEV_CLAMP を超えないよう切り詰める（サイクル長のずれの上限）。
    _raw = delta
    delta = clamp_delta_for_cycle(phase_idx, delta)
    if delta != _raw:
        # 切り詰めた分は累積に乗せない（実際に適用した量だけを累積する）
        j_cumulative[phase_idx] += delta - _raw
        _tag += f"[枠制限 {_raw:+d}→{delta:+d}]"

    j_applied_delta[phase_idx] = delta
    apply_j_logic()
    _cycle_dev = j_applied_delta[J_GREEN_PHASES[0]] + j_applied_delta[J_GREEN_PHASES[1]]
    print(f"✅ [道路{road_id}制御#{count}] {_tag} "
          f"pred[t3,t4]={np.round(np.asarray(t34), 1)} delta={delta} "
          f"累積={j_cumulative[phase_idx]}s サイクルずれ={_cycle_dev:+d}s (step={step})")
    return delta

# ============================================================
# 測定タイミング算出
# ============================================================
def get_measurement_times_for_main_road(sim_time, base_t1, base_t2, delta_adjustment=0):
    """
    道路1（主道路）の赤信号 前半/後半 測定時刻を返す。
    delta_adjustment = last_phase_delta（フェーズ0 の変化量）を渡す。
    フェーズ0が延長（delta>0）されると主道路の赤が短くなるため、
    測定タイミングをその分だけ前倒しする。
    """
    t1_time = int(sim_time + base_t1 + delta_adjustment // 2)
    t2_time = int(sim_time + base_t1 + base_t2 + delta_adjustment)
    return t1_time, t2_time


def get_measurement_times_for_minor_road(sim_time, base_t1, base_t2, delta_adjustment=0):
    """
    道路5/6（従道路）の赤信号 前半/後半 測定時刻を返す。
    delta_adjustment = last_phase_delta_road5（フェーズ6 の変化量）を渡す。
    フェーズ6が延長（delta>0）されると従道路の赤が短くなるため、
    測定タイミングをその分だけ前倒しする。
    """
    t1_time = int(sim_time + base_t1 + delta_adjustment // 2)
    t2_time = int(sim_time + base_t1 + base_t2 + delta_adjustment)
    return t1_time, t2_time


# ==== 時間距離図 =============================================================
import matplotlib
matplotlib.use("Agg")   # 非対話バックエンド: plt.show() でブロックさせない（バッチ実行用）
import matplotlib.pyplot as plt

# パラメータ
MEASURE_START = 30050  # 計測開始秒
MEASURE_END   = 31000  # 計測終了秒
MAX_MEASURE_VEHICLES = None  # Noneなら無制限、数値なら台数制限

DEFAULT_INTERVAL = 120
DEFAULT_MAX_N = 5

# ================================================================
# ★ ここだけ設定すればOK
# ================================================================

# 使用する .sumocfg または .net.xml のパス
NET_FILE_PATH = r"C:\Users\Tsukasa\Desktop\研究\予測\町モデルデータ\toyama_shouwa.sumocfg"

# 時間距離図に表示する信号交差点（南端→北端の順）
# (TL信号ID, 表示名 [, 赤フェーズmin, 赤フェーズmax])
# 赤フェーズを省略すると RED_PHASE_DEFAULT が使われる
TL_ORDER = [
    ("P", "P交差点"),
    ("O", "O交差点"),
    ("N", "N交差点"),
    ("M", "M交差点"),
    ("J", "J交差点", 2, 11),   # J交差点は独自フェーズ範囲
    ("F", "F交差点"),
    ("E", "E交差点"),
    ("D", "D交差点"),
    ("A", "A交差点", 2, 11),   # A交差点は独自フェーズ範囲
]

# TL_ORDER の代表TL以外に同じ交差点で監視する追加TL信号
# {表示名: [(TL_ID [, 赤フェーズmin, 赤フェーズmax]), ...]}
# 赤フェーズを省略すると RED_PHASE_DEFAULT が使われる
# いずれかの TL が赤なら「その交差点が赤」と判定する（OR条件）
EXTRA_RED_MONITORS = {}

# 赤信号と判定するデフォルトのフェーズ番号の範囲（min以上max以下）
RED_PHASE_DEFAULT = (2, 5)

# ================================================================
# net.xml 自動取得ロジック（変更不要）
# ================================================================

def _resolve_net_file(path):
    """sumocfg の場合はアクティブな net-file を取得して絶対パスを返す"""
    if path.endswith('.sumocfg'):
        base_dir = os.path.dirname(os.path.abspath(path))
        tree = ET.parse(path)
        root = tree.getroot()
        for net_elem in root.iter('net-file'):
            val = net_elem.get('value', '').strip()
            if val:
                return os.path.join(base_dir, val)
        raise FileNotFoundError(f"sumocfg に有効な net-file が見つかりません: {path}")
    return os.path.abspath(path)


def _build_tl_junction_map(net_xml_path):
    """
    connection 要素の via 属性から TL ID → junction node ID のマッピングを構築する。
    via 属性の形式は ':junctionID_接続番号_レーン番号'。
    """
    tree = ET.parse(net_xml_path)
    root = tree.getroot()
    tl_to_junc = {}
    for conn in root.findall('.//connection'):
        tl = conn.get('tl')
        via = conn.get('via')
        if tl and via:
            # ':junctionID_数字_数字' から junctionID を抽出（アンダースコア含む名前にも対応）
            m = re.match(r':(.+)_\d+_\d+$', via)
            if m:
                tl_to_junc[tl] = m.group(1)
    return tl_to_junc


def _bfs_between_nodes(net, from_node, to_node, max_hops=300):
    """
    from_node から to_node までの最短エッジ列を BFS で探索して返す。
    内部エッジ（交差点内）は sumolib が withInternal=False の場合除外済み。
    """
    from collections import deque

    target_id = to_node.getID()
    visited = {from_node.getID()}
    queue = deque()

    for edge in from_node.getOutgoing():
        nxt = edge.getToNode()
        if nxt.getID() not in visited:
            queue.append((nxt, [edge]))

    while queue:
        node, path = queue.popleft()
        node_id = node.getID()

        if node_id == target_id:
            return path

        if node_id in visited or len(path) >= max_hops:
            continue
        visited.add(node_id)

        for edge in node.getOutgoing():
            nxt_id = edge.getToNode().getID()
            if nxt_id not in visited:
                queue.append((edge.getToNode(), path + [edge]))

    return []


def setup_from_net(net_file_or_sumocfg, tl_order):
    """
    .net.xml を読み込み、指定した信号交差点順にルートを BFS で自動探索して
    エッジリスト・エッジ長・信号交差点位置を返す。

    Parameters
    ----------
    net_file_or_sumocfg : str
        .net.xml または .sumocfg のパス
    tl_order : list of (tl_id, label)
        南端→北端の順で信号交差点を指定。例: [("P","P交差点"), ..., ("A","A交差点")]

    Returns
    -------
    south_edges : list[str]
        南→北方向のエッジ ID 列
    north_edges : list[str]
        北→南方向のエッジ ID 列
    edge_lengths : dict[str, int]
        {エッジID: 長さ[m]}（四捨五入）
    signal_intersections : dict[str, int]
        {交差点名: 南端からの累積距離[m]}
    """
    net_file = _resolve_net_file(net_file_or_sumocfg)
    net = sumolib.net.readNet(net_file, withInternal=False)

    tl_junc_map = _build_tl_junction_map(net_file)

    # TL ID → sumolib Node
    tl_nodes = {}
    for entry in tl_order:
        tl_id, label = entry[0], entry[1]
        junc_id = tl_junc_map.get(tl_id)
        if junc_id is None:
            raise ValueError(f"TL '{tl_id}' に対応する junction が見つかりません（net.xml に tl='{tl_id}' の connection がない可能性があります）")
        try:
            tl_nodes[tl_id] = net.getNode(junc_id)
        except Exception:
            raise ValueError(f"TL '{tl_id}' の junction '{junc_id}' を sumolib で取得できませんでした")

    tl_id_list = [t[0] for t in tl_order]

    # --- 南→北ルート（P → A 方向）---
    south_edges_obj = []
    for i in range(len(tl_id_list) - 1):
        seg = _bfs_between_nodes(net, tl_nodes[tl_id_list[i]], tl_nodes[tl_id_list[i + 1]])
        if not seg:
            raise RuntimeError(
                f"{tl_id_list[i]}→{tl_id_list[i+1]} 間のルートが見つかりません。"
                "BFS の max_hops を増やすか、TL_ORDER を確認してください。"
            )
        south_edges_obj.extend(seg)

    # --- 北→南ルート（A → P 方向）---
    north_edges_obj = []
    north_segments = []  # [(seg_edges, from_tl_id, to_tl_id), ...]
    for i in range(len(tl_id_list) - 1, 0, -1):
        seg = _bfs_between_nodes(net, tl_nodes[tl_id_list[i]], tl_nodes[tl_id_list[i - 1]])
        if not seg:
            raise RuntimeError(
                f"{tl_id_list[i]}→{tl_id_list[i-1]} 間の逆ルートが見つかりません。"
                "BFS の max_hops を増やすか、TL_ORDER を確認してください。"
            )
        north_edges_obj.extend(seg)
        north_segments.append((seg, tl_id_list[i], tl_id_list[i - 1]))

    south_edges = [e.getID() for e in south_edges_obj]
    north_edges = [e.getID() for e in north_edges_obj]

    # エッジ長（最初のレーンの長さを使用）
    edge_lengths = {}
    for e in south_edges_obj + north_edges_obj:
        eid = e.getID()
        if eid not in edge_lengths:
            edge_lengths[eid] = round(e.getLanes()[0].getLength())

    # 各 TL 交差点の累積距離（南端 = 0m）
    junc_id_to_info = {tl_nodes[e[0]].getID(): (e[0], e[1]) for e in tl_order}
    # 起点交差点（南端 = 0m）を先に登録する
    # （起点は FROM ノードのため、エッジ探索ループには現れない）
    signal_intersections = {tl_order[0][1]: 0}
    cum = 0
    for edge in south_edges_obj:
        eid = edge.getID()
        cum += edge_lengths.get(eid, 0)
        to_junc_id = edge.getToNode().getID()
        if to_junc_id in junc_id_to_info:
            _, label = junc_id_to_info[to_junc_id]
            signal_intersections[label] = cum

    # --- 北方向エッジ → 南軸 X 座標マップ ---
    # 各北方向エッジの pos=0（A側）と pos=length（P側）の南軸 x を
    # TL 交差点位置をアンカーとして比例補間で算出する。
    tl_id_to_label = {e[0]: e[1] for e in tl_order}
    north_edge_x_map = {}  # {edge_id: (x_at_pos0, x_at_pos_end)}
    for seg_edges, from_tl, to_tl in north_segments:
        x_from = signal_intersections.get(tl_id_to_label.get(from_tl, ""), None)
        x_to   = signal_intersections.get(tl_id_to_label.get(to_tl,   ""), None)
        if x_from is None or x_to is None:
            continue
        total_seg_len = sum(edge_lengths.get(e.getID(), 0) for e in seg_edges)
        if total_seg_len == 0:
            continue
        cum = 0
        for edge in seg_edges:
            eid  = edge.getID()
            elen = edge_lengths.get(eid, 0)
            # pos=0 は A 側（x_from），pos=elen は P 側（x_to）に向かって線形補間
            x0 = x_from + (cum / total_seg_len) * (x_to - x_from)
            x1 = x_from + ((cum + elen) / total_seg_len) * (x_to - x_from)
            north_edge_x_map[eid] = (x0, x1)
            cum += elen

    print("=== net.xml から自動取得したルート情報 ===")
    print(f"南→北エッジ列: {south_edges}")
    print(f"北→南エッジ列: {north_edges}")
    print("エッジ長:")
    for eid in south_edges:
        print(f"  {eid}: {edge_lengths[eid]} m")
    print("信号交差点位置（南端からの累積距離）:")
    for name, pos in signal_intersections.items():
        print(f"  {name}: {pos} m")

    return south_edges, north_edges, edge_lengths, signal_intersections, north_edge_x_map


# net.xml からルート・距離・交差点位置を自動取得
(SOUTH_EDGES, NORTH_EDGES,
 EDGE_LENGTHS, SIGNAL_INTERSECTIONS,
 NORTH_EDGE_X_MAP) = setup_from_net(NET_FILE_PATH, TL_ORDER)

SOUTH_START_EDGE    = SOUTH_EDGES[0]   # 南方向の計測開始エッジ（P交差点直後）
SOUTH_COMPLETE_EDGE = SOUTH_EDGES[-1]  # 南方向の計測完了エッジ（A交差点通過後）
NORTH_START_EDGE    = NORTH_EDGES[0]   # 北方向の計測開始エッジ（A交差点直後）
NORTH_COMPLETE_EDGE = NORTH_EDGES[-1]  # 北方向の計測完了エッジ（P交差点通過後）

# 記録用
enter_time_south = {}
enter_time_north = {}
measured_south = set()
measured_north = set()

vehicle_traces = defaultdict(list)
completed_south = set()
completed_north = set()

# 信号赤時間記録用
signal_red_intervals = defaultdict(list)   # {交差点名: [[start, end], ...]}
red_active_states = {}                     # {交差点名: bool}


# ===============================
# ログ取得関数
# ===============================
def log_vehicle_positions(sim_time):
    global enter_time_south, enter_time_north
    global measured_south, measured_north
    global vehicle_traces, completed_south, completed_north
    global signal_red_intervals, red_active_states

    if sim_time < MEASURE_START or sim_time > MEASURE_END:
        return

    # === 各交差点の赤時間を監視 ===
    # TL_ORDER と EXTRA_RED_MONITORS を統合し、交差点ごとに一括判定する
    # (OR条件: いずれかの TL が赤なら「その交差点が赤」)
    intersection_red = {}

    # TL_ORDER の代表TLをチェック
    for entry in TL_ORDER:
        tl_id, name = entry[0], entry[1]
        red_min, red_max = entry[2:4] if len(entry) >= 4 else RED_PHASE_DEFAULT
        phase = traci.trafficlight.getPhase(tl_id)
        # まだ赤と判定されていない場合のみ上書き（OR条件）
        if name not in intersection_red:
            intersection_red[name] = False
        if red_min <= phase <= red_max:
            intersection_red[name] = True

    # EXTRA_RED_MONITORS の追加TLをチェック（OR条件で合算）
    for name, monitors in EXTRA_RED_MONITORS.items():
        if name not in intersection_red:
            intersection_red[name] = False
        for mon in monitors:
            tl_id = mon[0]
            red_min, red_max = (mon[1], mon[2]) if len(mon) >= 3 else RED_PHASE_DEFAULT
            phase = traci.trafficlight.getPhase(tl_id)
            if red_min <= phase <= red_max:
                intersection_red[name] = True
                break  # この交差点はもう赤確定

    # red_active_states を更新
    for name, red_now in intersection_red.items():
        if red_now:
            if not red_active_states.get(name, False):
                signal_red_intervals[name].append([sim_time, None])
                red_active_states[name] = True
        else:
            if red_active_states.get(name, False):
                if signal_red_intervals[name]:
                    signal_red_intervals[name][-1][1] = sim_time
                red_active_states[name] = False

    south_edge_set = set(SOUTH_EDGES)
    north_edge_set = set(NORTH_EDGES)

    # === 南北方向の検知・記録 ===
    for veh_id in traci.vehicle.getIDList():
        edge_id = traci.vehicle.getRoadID(veh_id)

        # 南方向: 起点エッジへの進入を検知
        if veh_id not in enter_time_south and edge_id == SOUTH_START_EDGE:
            if (MAX_MEASURE_VEHICLES is None) or (len(measured_south) < MAX_MEASURE_VEHICLES):
                enter_time_south[veh_id] = sim_time
                measured_south.add(veh_id)

        # 北方向: 起点エッジへの進入を検知
        if veh_id not in enter_time_north and edge_id == NORTH_START_EDGE:
            if (MAX_MEASURE_VEHICLES is None) or (len(measured_north) < MAX_MEASURE_VEHICLES):
                enter_time_north[veh_id] = sim_time
                measured_north.add(veh_id)

        # 南方向の記録
        if veh_id in measured_south and edge_id in south_edge_set:
            pos = traci.vehicle.getLanePosition(veh_id)
            vehicle_traces[veh_id].append(("south", sim_time, edge_id, pos))
            if edge_id == SOUTH_COMPLETE_EDGE:
                completed_south.add(veh_id)

        # 北方向の記録
        if veh_id in measured_north and edge_id in north_edge_set:
            pos = traci.vehicle.getLanePosition(veh_id)
            vehicle_traces[veh_id].append(("north", sim_time, edge_id, pos))
            if edge_id == NORTH_COMPLETE_EDGE:
                completed_north.add(veh_id)


# ===============================
# 座標変換
# ===============================
def convert_x_south(row):
    """南→北方向: 南端(0m)からの累積距離に変換"""
    x = 0
    for edge in SOUTH_EDGES:
        if row["edge"] == edge:
            return x + row["pos"]
        x += EDGE_LENGTHS[edge]
    return None

def convert_x_north(row):
    """北→南方向: 南軸（P=0m, A=総距離m）上の x 座標に変換する。
    TL 交差点位置をアンカーとした比例補間マップ NORTH_EDGE_X_MAP を使用する。"""
    params = NORTH_EDGE_X_MAP.get(row["edge"])
    if params is None:
        return None
    x0, x1 = params          # x0: pos=0(A側)の x, x1: pos=length(P側)の x
    elen = EDGE_LENGTHS.get(row["edge"], 0)
    if elen == 0:
        return x0
    return x0 + (row["pos"] / elen) * (x1 - x0)


# ===============================
# プロット関数
# ===============================
def plot_time_space_diagram():
    # === デバッグ診断 ===
    print(f"\n[DEBUG] measured_north の車両数: {len(measured_north)}")
    print(f"[DEBUG] NORTH_START_EDGE: {NORTH_START_EDGE}")
    print(f"[DEBUG] NORTH_EDGES (先頭5): {NORTH_EDGES[:5]}")
    print(f"[DEBUG] NORTH_EDGE_X_MAP エントリ数: {len(NORTH_EDGE_X_MAP)}")
    print(f"[DEBUG] NORTH_EDGE_X_MAP (先頭3): {list(NORTH_EDGE_X_MAP.items())[:3]}")
    for veh_id in list(measured_north)[:3]:
        records = [r for r in vehicle_traces[veh_id] if r[0] == "north"]
        edges_used = list(dict.fromkeys(r[2] for r in records))
        in_map = [e for e in edges_used if e in NORTH_EDGE_X_MAP]
        print(f"[DEBUG] 車両 {veh_id}: north記録={len(records)}件, エッジ={edges_used[:5]}, マップ一致={in_map[:5]}")
    # ==================

    plt.figure(figsize=(14, 7))

    def draw_traces(df, color):
        plt.plot(df["x"], df["sim_time"], color=color, alpha=0.4)

    # 南方向（P→A：A到達完了に限らず、P進入を検知した全車両をプロット）
    for veh_id in measured_south:
        if veh_id not in enter_time_south:
            continue
        records = [r for r in vehicle_traces[veh_id] if r[0] == "south"]
        if not records:
            continue
        df = pd.DataFrame(records, columns=["dir", "sim_time", "edge", "pos"])
        df["x"] = df.apply(convert_x_south, axis=1)
        df = df.dropna(subset=["x"])
        if df.empty:
            continue
        draw_traces(df, "blue")

    # 北方向（A→P：P到達完了に限らず、A進入を検知した全車両をプロット）
    for veh_id in measured_north:
        if veh_id not in enter_time_north:
            continue
        records = [r for r in vehicle_traces[veh_id] if r[0] == "north"]
        if not records:
            continue
        df = pd.DataFrame(records, columns=["dir", "sim_time", "edge", "pos"])
        df["x"] = df.apply(convert_x_north, axis=1)
        df = df.dropna(subset=["x"])
        if df.empty:
            continue
        draw_traces(df, "green")

    # 軸ラベル
    plt.xlabel("リンク距離 [m]")
    plt.ylabel("シミュレーション時間 [s]")
    plt.title("時間距離図（信号制御付き, シミュレーション時間基準）")

    # 主目盛り: 南端 → 各TL交差点 → 北端（TL_ORDER から自動生成）
    tick_positions = [0]
    tick_labels    = [TL_ORDER[0][1]]  # 南端 = P交差点
    for entry in TL_ORDER[1:]:
        label = entry[1]
        if label in SIGNAL_INTERSECTIONS:
            tick_positions.append(SIGNAL_INTERSECTIONS[label])
            tick_labels.append(label)
    total_south = sum(EDGE_LENGTHS[e] for e in SOUTH_EDGES)
    tick_positions.append(total_south)
    tick_labels.append("北端")

    plt.xticks(tick_positions, tick_labels)
    plt.grid(True, which="major", linestyle="--", alpha=0.6)

    # 赤信号時間帯の描画（sim_time 基準）
    for name, x_pos in SIGNAL_INTERSECTIONS.items():
        if name in signal_red_intervals:
            for start, end in signal_red_intervals[name]:
                if end is None:
                    end = MEASURE_END
                plt.fill_betweenx([start, end],
                                  x_pos - 5, x_pos + 5,
                                  color="red", alpha=0.3)

    # 凡例
    legend_lines = [
        plt.Line2D([0], [0], color="blue", lw=2, label="南→北"),
        plt.Line2D([0], [0], color="green", lw=2, label="北→南"),
        plt.Rectangle((0, 0), 1, 1, color="red", alpha=0.3, label="赤信号")
    ]
    plt.legend(handles=legend_lines)

    plt.show()
# ============================================================================
# ============================================================================


# ============================================================
# ギャップ感応制御クラス（J交差点 主道路/従道路 両対応）
# CONTROL_MODE == "gap" のときに run_simulation 内でインスタンス化する。
# ============================================================
# setPhaseDuration は「次のステップから」効くため、青の開始を検知した時点で既に1秒経過しており、
# 「43秒」と指示すると実際には 44秒 光ってしまう（実測で確認）。
# ★指令する残り時間から この1秒を引いて、実際の青が指示どおりの長さになるようにする。
#   偏差の計算は「実測の青 − BASE」をそのまま使う（帳簿ではなく実物を合わせる）。
#   ※ 以前は偏差の側から1秒引いていたため、累積が毎サイクル1秒ぶん余計にマイナスへ振れ、
#     押し戻しの調整サイクルが過剰に入って実際の青が BASE より約1秒長くなっていた。
GREEN_STEP_OFFSET = 1


class GapCycleController:
    def __init__(self, tls_id):
        self.tls_id = tls_id

        self.MAIN_GREEN  = 0   # 主道路(道路1)青フェーズ番号
        self.MINOR_GREEN = 5   # 従道路(道路5)青フェーズ番号（現ネット: phase5=道路5緑。旧net phase6は誤り）

        # ── 青時間の基準と可動範囲 ──────────────────
        # ★以前は BASE/MIN/MAX を 60/50/70（主）・26/16/36（従）で固定していたため、
        #   時間帯別サイクル長（100/110/130s）が変わっても J だけ青が追従せず、
        #   他の信号（A〜P）とサイクル長が食い違っていた。
        #   現在は「その時間帯のスケジュール青 ± J_GREEN_DEV_CLAMP」を基準にし、
        #   さらにサイクル長のずれが ±J_CYCLE_DEV_CLAMP に収まるよう毎青ごとに
        #   _refresh_bounds() で取り直す（予測制御と同じ枠）。
        # ※ 派生クラス（pgapmin/npred）が MIN/MAX を独自に設定する場合は
        #   self.follow_schedule = False にすればこの追従を止められる。
        self.follow_schedule = True
        self.MIN_MAIN_GREEN = self.MAX_MAIN_GREEN = self.BASE_MAIN_GREEN = 0.0
        self.MIN_MINOR_GREEN = self.MAX_MINOR_GREEN = self.BASE_MINOR_GREEN = 0.0
        self._refresh_bounds()
        self.MIN_MAIN_GREEN_2  = self.MIN_MAIN_GREEN
        self.MIN_MINOR_GREEN_2 = self.MIN_MINOR_GREEN

        # 直近の青が「スケジュール青からどれだけずれたか」（サイクル長の枠計算に使う）
        self.last_dev = {self.MAIN_GREEN: 0, self.MINOR_GREEN: 0}
        # 現在の青に割り当てている長さ（MIN から始めて、車が来るたび延長する）
        self.main_allotted  = 0.0
        self.minor_allotted = 0.0
        # その青を開始した時点の BASE（= スケジュール青）。偏差はこれを基準に測る。
        # ※ 時間帯別サイクル長の切替が青の途中で起きると BASE が変わるため、
        #   開始時の値を控えておかないと「割り当てたときの基準」と「測るときの基準」が
        #   食い違い、偏差が枠(±10s)を超える（実測で +25s の例あり）。
        self.main_base_at_start  = 0.0
        self.minor_base_at_start = 0.0

        # ── 共通 ────────────────────────────────────
        self.MAX_GAP = 5

        # ── 感知器（ループコイル） ──────────────────
        self.MAJOR_DETECTORS = ["J_south_1"]
        self.MINOR_DETECTORS = ["J_west_0"]

        # ── 内部状態：主道路 ────────────────────────
        self.main_green_active       = False
        self.main_green_start_time   = None
        self.last_major_vehicle_time = 0
        self.main_cumulative         = 0     # 累積制御偏差（60sベースからの偏差合計）
        self.main_adj_pending        = False # 調整サイクル予約フラグ
        self.main_adj_active         = False # 調整サイクル実行中フラグ
        # ── 内部状態：従道路 ────────────────────────
        self.minor_green_active       = False
        self.minor_green_start_time   = None
        self.last_minor_vehicle_time  = 0
        self.minor_cumulative         = 0
        self.minor_adj_pending        = False
        self.minor_adj_active         = False
        # SUMO側の青時間は上限(MAX)に設定しておき、実際の長さは
        # 「MIN以上でギャップ発生 / MAXで強制打ち切り」で決める（通常サイクル終了時も同じ）。
        # ※ 以前は既定パスで BASE を一度設定するだけだったため、時間帯別サイクル長が
        #   変わっても SUMO 側の duration が初期値のまま残り、青が上限まで伸びなかった。
        self._set_phase_duration(self.MAIN_GREEN, self.MAX_MAIN_GREEN)
        self._set_phase_duration(self.MINOR_GREEN, self.MAX_MINOR_GREEN)

    def _refresh_bounds(self):
        """その時間帯のスケジュール青を基準に、青の下限/上限を取り直す。
        主道路と従道路それぞれ「スケジュール青 ±10s」かつ「両者の合計ずれが ±10s以内」。
        目標サイクル長が変わったとき・各青の開始時に呼ぶ。"""
        if not getattr(self, "follow_schedule", True):
            return
        _last = getattr(self, "last_dev", {self.MAIN_GREEN: 0, self.MINOR_GREEN: 0})
        lo, hi = green_bounds_for_phase(self.MAIN_GREEN, _last.get(self.MINOR_GREEN, 0))
        self.BASE_MAIN_GREEN = j_scheduled_green[self.MAIN_GREEN]
        self.MIN_MAIN_GREEN, self.MAX_MAIN_GREEN = lo, hi
        lo, hi = green_bounds_for_phase(self.MINOR_GREEN, _last.get(self.MAIN_GREEN, 0))
        self.BASE_MINOR_GREEN = j_scheduled_green[self.MINOR_GREEN]
        self.MIN_MINOR_GREEN, self.MAX_MINOR_GREEN = lo, hi

    def _apply_max_durations(self):
        """サイクル長が変わったとき、SUMO側の青時間を新しい上限(MAX)に合わせ直す。
        調整サイクルの予約中/実行中は adj_dur を壊さないよう触らない。"""
        if not getattr(self, "follow_schedule", True):
            return
        if not (self.main_adj_pending or self.main_adj_active):
            self._set_phase_duration(self.MAIN_GREEN, self.MAX_MAIN_GREEN)
        if not (self.minor_adj_pending or self.minor_adj_active):
            self._set_phase_duration(self.MINOR_GREEN, self.MAX_MINOR_GREEN)

    def _set_phase_duration(self, phase_index, duration):
        from traci._trafficlight import Phase
        logic  = traci.trafficlight.getAllProgramLogics(self.tls_id)[0]
        phases = list(logic.getPhases())
        p = phases[phase_index]
        phases[phase_index] = Phase(duration=duration, state=p.state,
                                    minDur=duration, maxDur=duration)
        new_logic = traci.trafficlight.Logic(
            programID=logic.programID, type=logic.type,
            currentPhaseIndex=logic.currentPhaseIndex, phases=phases)
        traci.trafficlight.setProgramLogic(self.tls_id, new_logic)

    def _vehicle_detected(self, detectors):
        """このステップに感知器上へ車が居たか（延長判定用）。"""
        for det in detectors:
            try:
                if traci.inductionloop.getLastStepVehicleNumber(det) > 0:
                    return True
            except Exception:
                pass
        return False

    def _start_green(self, phase_index, allotted):
        """青の開始時に、その青へ割り当てる長さをSUMOに与える。
        setPhaseDuration は次ステップから効くので、指令値から GREEN_STEP_OFFSET を引いて
        実際の青が allotted ちょうどになるようにする。"""
        try:
            traci.trafficlight.setPhaseDuration(
                self.tls_id, max(0.0, allotted - GREEN_STEP_OFFSET))
        except Exception as e:
            print(f"⚠️ setPhaseDuration 失敗(phase={phase_index}): {e}")
        return allotted

    def _extend_green(self, green_time, allotted, max_green):
        """車が来たので「今から MAX_GAP 秒先」まで延長する（上限 max_green）。
        延長後の割当を返す。延長不要ならそのまま返す。"""
        new_end = min(green_time + self.MAX_GAP, max_green)
        if new_end > allotted:
            try:
                traci.trafficlight.setPhaseDuration(
                    self.tls_id, max(0.0, new_end - green_time - GREEN_STEP_OFFSET))
            except Exception as e:
                print(f"⚠️ setPhaseDuration(延長) 失敗: {e}")
            return new_end
        return allotted

    def _gap_occurred_major(self, step):
        for det in self.MAJOR_DETECTORS:
            try:
                if traci.inductionloop.getLastStepVehicleNumber(det) > 0:
                    self.last_major_vehicle_time = step
                    return False
            except Exception:
                pass
        return (step - self.last_major_vehicle_time) >= self.MAX_GAP

    def _gap_occurred_minor(self, step):
        for det in self.MINOR_DETECTORS:
            try:
                if traci.inductionloop.getLastStepVehicleNumber(det) > 0:
                    self.last_minor_vehicle_time = step
                    return False
            except Exception:
                pass
        return (step - self.last_minor_vehicle_time) >= self.MAX_GAP

    def update(self, step):
        phase = traci.trafficlight.getPhase(self.tls_id)

        # ── 主道路「青」開始検知 ──────────────────────
        if phase == self.MAIN_GREEN and not self.main_green_active:
            self.main_green_active       = True
            self.main_green_start_time   = step
            self.last_major_vehicle_time = step
            self._refresh_bounds()   # その時間帯のスケジュール青 ±10s（かつサイクル ±10s）に更新
            self.main_base_at_start = self.BASE_MAIN_GREEN   # 偏差の基準をこの青の開始時点で固定
            if self.main_adj_pending:
                self.main_adj_active  = True
                self.main_adj_pending = False
                # 調整サイクル: いま有効な BASE を基準に ±OFFSET_ADJUST_PER_CYCLE を割り当てる。
                # （前サイクル終了時の値を使い回すと、間にサイクル長切替が入ったとき
                #   基準がずれて偏差が枠を外れる）
                _t = self.BASE_MAIN_GREEN + (-OFFSET_ADJUST_PER_CYCLE if self.main_cumulative > 0
                                    else OFFSET_ADJUST_PER_CYCLE)
                self.main_allotted = self._start_green(
                    self.MAIN_GREEN, max(self.MIN_MAIN_GREEN, min(self.MAX_MAIN_GREEN, _t)))
                print(f"🔄 [GAP主道路・調整開始] 割当={self.main_allotted:.0f}s (step={step})")
            else:
                # 通常サイクル: MIN（= BASE−10s）から開始し、車が来るたびに延長する
                self.main_allotted = self._start_green(self.MAIN_GREEN, self.MIN_MAIN_GREEN)

        # ── 主道路「青」終了検知 ──────────────────────
        elif phase != self.MAIN_GREEN and self.main_green_active:
            green_time             = step - self.main_green_start_time
            self.main_green_active = False
            deviation = green_time - self.main_base_at_start
            self.last_dev[self.MAIN_GREEN] = deviation   # 従道路側の枠計算に使う

            # ★累積オフセット補正は常時有効（予測制御と同じ方式に揃えた）。
            #   以前は ENABLE_EVEN_CYCLE_CANCEL=False のとき偏差を表示するだけで
            #   累積も調整サイクルも動かず、サイクル長のずれが押し戻されなかった。
            if step < PREP_TIME:
                # 準備時間中は累積しない。開始直後の1本目の青は traci 接続時に
                # 既に進行中だった位相の残りなので偏差が過大に出て累積を汚すため。
                # （予測制御も準備時間後にしか動かないので、これで条件が揃う）
                self._set_phase_duration(self.MAIN_GREEN, self.MAX_MAIN_GREEN)
            elif self.main_adj_active:
                # 調整サイクル終了
                self.main_adj_active  = False
                self.main_cumulative += deviation
                print(f"✅ [GAP主道路・調整終了] G={int(green_time)}s, 累積={self.main_cumulative}s (step={int(step)})")
                # 1回の補正で足りない場合は追加調整サイクルを予約
                if abs(self.main_cumulative) >= OFFSET_ADJUST_PER_CYCLE:
                    adj_dur = max(self.MIN_MAIN_GREEN, min(self.MAX_MAIN_GREEN,
                                  self.BASE_MAIN_GREEN + (-OFFSET_ADJUST_PER_CYCLE if self.main_cumulative > 0 else OFFSET_ADJUST_PER_CYCLE)))
                    self._set_phase_duration(self.MAIN_GREEN, adj_dur)
                    self.main_adj_pending = True
                    print(f"🔔 [GAP主道路] 累積={self.main_cumulative}s → 追加調整サイクル予定 ({adj_dur}s)")
                else:
                    self._set_phase_duration(self.MAIN_GREEN, self.MAX_MAIN_GREEN)
            else:
                # 通常サイクル終了
                self.main_cumulative += deviation
                print(f"🔀 [GAP主道路] G={int(green_time)}s, 偏差={int(deviation):+d}s, 累積={self.main_cumulative}s (step={int(step)})")
                if abs(self.main_cumulative) >= OFFSET_ADJUST_THRESHOLD:
                    adj_dur = max(self.MIN_MAIN_GREEN, min(self.MAX_MAIN_GREEN,
                                  self.BASE_MAIN_GREEN + (-OFFSET_ADJUST_PER_CYCLE if self.main_cumulative > 0 else OFFSET_ADJUST_PER_CYCLE)))
                    self._set_phase_duration(self.MAIN_GREEN, adj_dur)
                    self.main_adj_pending = True
                    print(f"🔔 [GAP主道路] 累積={self.main_cumulative}s → 次サイクル調整予定 ({adj_dur}s)")
                else:
                    self._set_phase_duration(self.MAIN_GREEN, self.MAX_MAIN_GREEN)

        # ── 従道路「青」開始検知 ──────────────────────
        if phase == self.MINOR_GREEN and not self.minor_green_active:
            self.minor_green_active       = True
            self.minor_green_start_time   = step
            self.last_minor_vehicle_time  = step
            self._refresh_bounds()   # その時間帯のスケジュール青 ±10s（かつサイクル ±10s）に更新
            self.minor_base_at_start = self.BASE_MINOR_GREEN   # 偏差の基準をこの青の開始時点で固定
            if self.minor_adj_pending:
                self.minor_adj_active  = True
                self.minor_adj_pending = False
                # 調整サイクル: いま有効な BASE を基準に ±OFFSET_ADJUST_PER_CYCLE を割り当てる。
                # （前サイクル終了時の値を使い回すと、間にサイクル長切替が入ったとき
                #   基準がずれて偏差が枠を外れる）
                _t = self.BASE_MINOR_GREEN + (-OFFSET_ADJUST_PER_CYCLE if self.minor_cumulative > 0
                                    else OFFSET_ADJUST_PER_CYCLE)
                self.minor_allotted = self._start_green(
                    self.MINOR_GREEN, max(self.MIN_MINOR_GREEN, min(self.MAX_MINOR_GREEN, _t)))
                print(f"🔄 [GAP従道路・調整開始] 割当={self.minor_allotted:.0f}s (step={step})")
            else:
                # 通常サイクル: MIN（= BASE−10s）から開始し、車が来るたびに延長する
                self.minor_allotted = self._start_green(self.MINOR_GREEN, self.MIN_MINOR_GREEN)

        # ── 従道路「青」終了検知 ──────────────────────
        elif phase != self.MINOR_GREEN and self.minor_green_active:
            green_time              = step - self.minor_green_start_time
            self.minor_green_active = False
            deviation = green_time - self.minor_base_at_start
            self.last_dev[self.MINOR_GREEN] = deviation   # 主道路側の枠計算に使う

            # ★累積オフセット補正は常時有効（主道路と同じ）
            if step < PREP_TIME:
                # 準備時間中は累積しない。開始直後の1本目の青は traci 接続時に
                # 既に進行中だった位相の残りなので偏差が過大に出て累積を汚すため。
                # （予測制御も準備時間後にしか動かないので、これで条件が揃う）
                self._set_phase_duration(self.MINOR_GREEN, self.MAX_MINOR_GREEN)
            elif self.minor_adj_active:
                # 調整サイクル終了
                self.minor_adj_active  = False
                self.minor_cumulative += deviation
                print(f"✅ [GAP従道路・調整終了] G={int(green_time)}s, 累積={self.minor_cumulative}s (step={int(step)})")
                # 1回の補正で足りない場合は追加調整サイクルを予約
                if abs(self.minor_cumulative) >= OFFSET_ADJUST_PER_CYCLE:
                    adj_dur = max(self.MIN_MINOR_GREEN, min(self.MAX_MINOR_GREEN,
                                  self.BASE_MINOR_GREEN + (-OFFSET_ADJUST_PER_CYCLE if self.minor_cumulative > 0 else OFFSET_ADJUST_PER_CYCLE)))
                    self._set_phase_duration(self.MINOR_GREEN, adj_dur)
                    self.minor_adj_pending = True
                    print(f"🔔 [GAP従道路] 累積={self.minor_cumulative}s → 追加調整サイクル予定 ({adj_dur}s)")
                else:
                    self._set_phase_duration(self.MINOR_GREEN, self.MAX_MINOR_GREEN)
            else:
                # 通常サイクル終了
                self.minor_cumulative += deviation
                print(f"🔀 [GAP従道路] G={int(green_time)}s, 偏差={int(deviation):+d}s, 累積={self.minor_cumulative}s (step={int(step)})")
                if abs(self.minor_cumulative) >= OFFSET_ADJUST_THRESHOLD:
                    adj_dur = max(self.MIN_MINOR_GREEN, min(self.MAX_MINOR_GREEN,
                                  self.BASE_MINOR_GREEN + (-OFFSET_ADJUST_PER_CYCLE if self.minor_cumulative > 0 else OFFSET_ADJUST_PER_CYCLE)))
                    self._set_phase_duration(self.MINOR_GREEN, adj_dur)
                    self.minor_adj_pending = True
                    print(f"🔔 [GAP従道路] 累積={self.minor_cumulative}s → 次サイクル調整予定 ({adj_dur}s)")
                else:
                    self._set_phase_duration(self.MINOR_GREEN, self.MAX_MINOR_GREEN)

        # ── 主道路「青」中: MIN から始めて、車が来るたびに延長 ─────────
        #   ・割当の初期値 = MIN（= BASE−10s）
        #   ・感知器に車が来たら「今から MAX_GAP(5s) 先」まで割当を伸ばす（上限 MAX = BASE+10s）
        #     → 5秒の隙間が空いた時点で割当が尽きて青が終わる（＝ギャップ感応）
        #   ・延長幅は最大 20s（MIN → MAX）で、予測制御の delta ±10s と同じ範囲に収まる
        #   ・MAX 到達時は保険として強制的に切る（調整サイクル中も枠を超えさせない）
        if phase == self.MAIN_GREEN and self.main_green_active:
            green_time = step - self.main_green_start_time
            if green_time >= self.MAX_MAIN_GREEN:
                traci.trafficlight.setPhase(self.tls_id, phase + 1)
            elif not self.main_adj_active:
                if self._vehicle_detected(self.MAJOR_DETECTORS):
                    self.last_major_vehicle_time = step
                    self.main_allotted = self._extend_green(
                        green_time, self.main_allotted, self.MAX_MAIN_GREEN)

        # ── 従道路「青」中: 同じく MIN から延長 ─────────
        if phase == self.MINOR_GREEN and self.minor_green_active:
            green_time = step - self.minor_green_start_time
            if green_time >= self.MAX_MINOR_GREEN:
                traci.trafficlight.setPhase(self.tls_id, phase + 1)
            elif not self.minor_adj_active:
                if self._vehicle_detected(self.MINOR_DETECTORS):
                    self.last_minor_vehicle_time = step
                    self.minor_allotted = self._extend_green(
                        green_time, self.minor_allotted, self.MAX_MINOR_GREEN)


# ============================================================
# 予測ギャップ感応制御（ギャップ骨格 + 到着間隔モデルNで判定を段階化）
# ------------------------------------------------------------
# ・骨格は GapCycleController のまま（検知器のギャップで青を延長/短縮）。
# ・到着間隔モデルで各青の N(最大間隔発生台数目) を予測し、
#   「通過台数 ≥ N になったらギャップ閾値を MAX_GAP→GAP_SHORT に緩める」だけを追加。
#   → 予測した自然な切れ目に来たら早めに切れる＝無駄青を削減。
#   → 予測が外れても、実ギャップが来ない限り切らない（＝従来ギャップに縮退）フェイルセーフ。
# ・N は 1車線検知器の通過台数と同じ「1車線あたり行数」単位（モデル側で÷3済み）なので
#   通過台数(1車線) ≥ N をそのまま比較する。
# ============================================================
class PredictiveGapController(GapCycleController):
    def __init__(self, tls_id, tracker, edge_index):
        super().__init__(tls_id)
        self.tracker = tracker
        self.edge_index = edge_index
        self.GAP_SHORT = PGAP_GAP_SHORT   # N到達後の緩めたギャップ閾値(秒)。R3_GAPSHORTで可変
        # 到着間隔モデル（道路1=主, 道路5=従）を事前ロード
        self.dual = {}
        for r in ("1", "5"):
            m, sw, si = arr.load_dual_model(r)
            self.dual[r] = (m, sw, si, arr.DUAL_ROADS[r]["node"])
        # 予測N と 青中の通過台数
        self.N_main = None
        self.N_minor = None
        self.passed_main = 0
        self.passed_minor = 0
        # 通過カウント用の1車線検知器（ギャップ検知と同じもの）
        self.MAIN_COUNT_DET  = self.MAJOR_DETECTORS[0]   # 主道路(道路1)
        self.MINOR_COUNT_DET = self.MINOR_DETECTORS[0]   # 従道路(道路5)
        # === 予測が生きた計測 ===
        #   pred_fire  : N予測が有効に出た青の回数（データ揃い）
        #   pred_effective: 通過台数≥Nで閾値を2sに緩めた結果、5sを待たずに切れた回数
        #   saved_sec  : そのとき削減できた無駄青の合計秒数（MAX_GAP−実ギャップ）
        self.pred_fire      = {"main": 0, "minor": 0}
        self.pred_effective = {"main": 0, "minor": 0}
        self.saved_sec      = {"main": 0.0, "minor": 0.0}
        self._eff_counted   = {"main": False, "minor": False}  # 1青あたり1回のみ計上

    def _predict_N(self, road_id, node_index):
        """全9道路の待ち台数(分割合計=1サイクル総数) + 対象道路の現在Nから、次サイクルNを予測。
        トラッカーが未整備(None)なら None を返す（→従来ギャップに縮退）。"""
        w9x2 = self.tracker.model_input()
        if w9x2 is None:
            return None
        wait_9 = w9x2.sum(axis=1)                       # 各道路の待ち台数合計 [9]
        n_cur = self.tracker.latest_N.get(road_id, 0)   # 現サイクルの実測N
        model, sw, si, node = self.dual[road_id]
        _W, N_pred = arr.predict_WN(wait_9, n_cur, model, sw, si, self.edge_index, node)
        return max(1, int(round(N_pred)))

    # ギャップ閾値を動的に: 通過台数がN以上なら GAP_SHORT、未満なら MAX_GAP
    def _gap_occurred_major(self, step):
        for det in self.MAJOR_DETECTORS:
            try:
                if traci.inductionloop.getLastStepVehicleNumber(det) > 0:
                    self.last_major_vehicle_time = step
                    return False
            except Exception:
                pass
        gap = step - self.last_major_vehicle_time
        relaxed = (self.N_main is not None and self.passed_main >= self.N_main)
        thr = self.GAP_SHORT if relaxed else self.MAX_GAP
        if gap >= thr:
            # 緩めた閾値(2s)で、5sを待たずに切れた＝予測が生きた
            if relaxed and gap < self.MAX_GAP and not self._eff_counted["main"]:
                self.pred_effective["main"] += 1
                self.saved_sec["main"] += (self.MAX_GAP - gap)
                self._eff_counted["main"] = True
            return True
        return False

    def _gap_occurred_minor(self, step):
        for det in self.MINOR_DETECTORS:
            try:
                if traci.inductionloop.getLastStepVehicleNumber(det) > 0:
                    self.last_minor_vehicle_time = step
                    return False
            except Exception:
                pass
        gap = step - self.last_minor_vehicle_time
        relaxed = (self.N_minor is not None and self.passed_minor >= self.N_minor)
        thr = self.GAP_SHORT if relaxed else self.MAX_GAP
        if gap >= thr:
            if relaxed and gap < self.MAX_GAP and not self._eff_counted["minor"]:
                self.pred_effective["minor"] += 1
                self.saved_sec["minor"] += (self.MAX_GAP - gap)
                self._eff_counted["minor"] = True
            return True
        return False

    def update(self, step):
        phase = traci.trafficlight.getPhase(self.tls_id)
        # 青開始時: N予測 + 通過カウントのリセット
        if phase == self.MAIN_GREEN and not self.main_green_active:
            self.N_main = self._predict_N("1", 0)
            self.passed_main = 0
            self._eff_counted["main"] = False
            if self.N_main is not None:
                self.pred_fire["main"] += 1
        if phase == self.MINOR_GREEN and not self.minor_green_active:
            self.N_minor = self._predict_N("5", 4)
            self.passed_minor = 0
            self._eff_counted["minor"] = False
            if self.N_minor is not None:
                self.pred_fire["minor"] += 1
        # 青中: 1車線検知器の通過台数を積算
        if phase == self.MAIN_GREEN:
            try:
                self.passed_main += traci.inductionloop.getLastStepVehicleNumber(self.MAIN_COUNT_DET)
            except Exception:
                pass
        if phase == self.MINOR_GREEN:
            try:
                self.passed_minor += traci.inductionloop.getLastStepVehicleNumber(self.MINOR_COUNT_DET)
            except Exception:
                pass
        # 骨格のギャップ制御（_gap_occurred_* は上でN連動にオーバーライド済み）
        super().update(step)

    def dump_stats(self, out_dir):
        """予測が生きた回数・削減無駄青をCSVに出力する。"""
        path = os.path.join(out_dir, "予測ギャップ_予測有効ログ.csv")
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["対象", "N予測発火(青)回数", "予測有効(早切り)回数",
                        "有効率(%)", "削減無駄青合計(秒)", "平均削減(秒/有効)"])
            for k, lbl in [("main", "主道路(道路1)"), ("minor", "従道路(道路5)")]:
                fire = self.pred_fire[k]; eff = self.pred_effective[k]; sav = self.saved_sec[k]
                rate = round(eff / fire * 100, 1) if fire else 0
                avg = round(sav / eff, 2) if eff else 0
                w.writerow([lbl, fire, eff, rate, round(sav, 1), avg])
            # 合計
            f_all = self.pred_fire["main"] + self.pred_fire["minor"]
            e_all = self.pred_effective["main"] + self.pred_effective["minor"]
            s_all = self.saved_sec["main"] + self.saved_sec["minor"]
            w.writerow(["合計", f_all, e_all,
                        round(e_all / f_all * 100, 1) if f_all else 0,
                        round(s_all, 1), round(s_all / e_all, 2) if e_all else 0])
        print(f"✅ 予測ギャップ 予測有効ログ: {path}")
        print(f"   [合計] N予測発火={f_all}回 / 予測有効(早切り)={e_all}回 "
              f"({round(e_all/f_all*100,1) if f_all else 0}%) / 削減無駄青={s_all:.0f}s")


# ============================================================
# N単独制御（ギャップ完全無視・到着間隔モデルNのみで青を切る）
# ------------------------------------------------------------
# ・予測ギャップ(PredictiveGap)の「N予測＋通過台数カウント」機構は流用するが、
#   ギャップ判定(検知器の無車間)を完全に外し、切る条件を
#     「青中の通過台数(1車線) ≥ 予測N」だけにする。
#   → N台の車群を捌いたら即青終了。ギャップは一切見ない。
# ・N は最大到着間隔が発生する台数目（車群の切れ目）。÷3済みで1車線通過台数と同単位。
# ・安全のため MIN/MAX 青の枠は残す（N が青長を決められるよう MIN は低め）。
#   N未定(予測不成立)の青は BASE 青で切る（フェイルセーフ）。
# ・env: R3_NMIN_MAIN/R3_NMIN_MINOR（青下限）, R3_NMAX_MAIN/R3_NMAX_MINOR（青上限）
# ============================================================
class PredNOnlyController(PredictiveGapController):
    def __init__(self, tls_id, tracker, edge_index):
        super().__init__(tls_id, tracker, edge_index)
        # ギャップを見ない分、MIN を下げて N が青長を決められるようにする。
        # 独自の MIN/MAX を使うので、スケジュール青への自動追従は止める。
        self.follow_schedule = False
        self.MIN_MAIN_GREEN  = int(os.environ.get("R3_NMIN_MAIN",  "15"))
        self.MAX_MAIN_GREEN  = int(os.environ.get("R3_NMAX_MAIN",  str(self.MAX_MAIN_GREEN)))
        self.MIN_MINOR_GREEN = int(os.environ.get("R3_NMIN_MINOR", "8"))
        self.MAX_MINOR_GREEN = int(os.environ.get("R3_NMAX_MINOR", str(self.MAX_MINOR_GREEN)))
        # 統計: 切り理由（N到達 / BASE(N未定) / MAX上限）と青時間
        self.cut_reason = {"main": {"N": 0, "BASE": 0, "MAX": 0},
                           "minor": {"N": 0, "BASE": 0, "MAX": 0}}
        self.green_sum  = {"main": 0.0, "minor": 0.0}
        self.green_cnt  = {"main": 0, "minor": 0}

    # ── ギャップ無視: 「通過台数 ≥ N」だけで切る（N未定は BASE 青で切る）──
    def _gap_occurred_major(self, step):
        gt = step - self.main_green_start_time
        if self.N_main is None:
            return gt >= self.BASE_MAIN_GREEN
        return self.passed_main >= self.N_main

    def _gap_occurred_minor(self, step):
        gt = step - self.minor_green_start_time
        if self.N_minor is None:
            return gt >= self.BASE_MINOR_GREEN
        return self.passed_minor >= self.N_minor

    def update(self, step):
        # 青終了の瞬間に青時間・切り理由を記録（親のupdateが状態を変える前に判定）
        phase = traci.trafficlight.getPhase(self.tls_id)
        if phase != self.MAIN_GREEN and self.main_green_active:
            gt = step - self.main_green_start_time
            self.green_sum["main"] += gt; self.green_cnt["main"] += 1
            if gt >= self.MAX_MAIN_GREEN:            self.cut_reason["main"]["MAX"]  += 1
            elif self.N_main is None:                self.cut_reason["main"]["BASE"] += 1
            else:                                    self.cut_reason["main"]["N"]    += 1
        if phase != self.MINOR_GREEN and self.minor_green_active:
            gt = step - self.minor_green_start_time
            self.green_sum["minor"] += gt; self.green_cnt["minor"] += 1
            if gt >= self.MAX_MINOR_GREEN:           self.cut_reason["minor"]["MAX"]  += 1
            elif self.N_minor is None:               self.cut_reason["minor"]["BASE"] += 1
            else:                                    self.cut_reason["minor"]["N"]    += 1
        super().update(step)

    def dump_stats(self, out_dir):
        path = os.path.join(out_dir, "N単独制御_ログ.csv")
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["対象", "青回数", "平均青(s)",
                        "N到達で切り", "BASE切り(N未定)", "MAX上限で切り",
                        "MIN青(s)", "MAX青(s)"])
            for k, lbl, mn, mx in [("main", "主道路(道路1)", self.MIN_MAIN_GREEN, self.MAX_MAIN_GREEN),
                                   ("minor", "従道路(道路5)", self.MIN_MINOR_GREEN, self.MAX_MINOR_GREEN)]:
                cnt = self.green_cnt[k]
                avg = round(self.green_sum[k] / cnt, 1) if cnt else 0
                cr = self.cut_reason[k]
                w.writerow([lbl, cnt, avg, cr["N"], cr["BASE"], cr["MAX"], mn, mx])
        print(f"✅ N単独制御ログ: {path}")
        for k, lbl in [("main", "主道路"), ("minor", "従道路")]:
            cnt = self.green_cnt[k]; avg = round(self.green_sum[k] / cnt, 1) if cnt else 0
            cr = self.cut_reason[k]
            print(f"   [{lbl}] 平均青={avg}s / N切り={cr['N']} BASE切り={cr['BASE']} MAX切り={cr['MAX']}")


# ============================================================
# 予測ギャップ感応制御【MIN調整版】(改善案①)
# ------------------------------------------------------------
# ・ギャップ判定は従来どおり 5秒のまま（車群の途中で切らない＝積み残しなし）。
# ・到着間隔モデルの W(次サイクル待ち台数) を各青開始で予測し、
#   需要が小さいほど MIN_GREEN を下げる。
#   → 低需要時に「待ち行列が捌けているのに MIN まで青を保持する無駄青」を削減。
#   → 5秒窓は触らないのでフェイルセーフ（積み残し増なし）。
#     MIN_GREEN = clamp(MIN_FLOOR, MIN_FLOOR+(BASE_MIN-MIN_FLOOR)*min(1, W/W_REF), BASE_MIN)
# ============================================================
class MinGreenPredGapController(GapCycleController):
    def __init__(self, tls_id, tracker, edge_index):
        super().__init__(tls_id)
        self.tracker = tracker
        self.edge_index = edge_index
        self.dual = {}
        for r in ("1", "5"):
            m, sw, si = arr.load_dual_model(r)
            self.dual[r] = (m, sw, si, arr.DUAL_ROADS[r]["node"])
        # MIN_GREEN 調整パラメータ（R3_MINFLOOR_* / R3_WREF_* で可変）
        # MIN を予測Wで動かす方式なので、スケジュール青への自動追従は止める。
        self.follow_schedule = False
        self.BASE_MIN  = {"main": self.MIN_MAIN_GREEN, "minor": self.MIN_MINOR_GREEN}
        self.MIN_FLOOR = {"main":  int(os.environ.get("R3_MINFLOOR_MAIN", "15")),
                          "minor": int(os.environ.get("R3_MINFLOOR_MINOR", "8"))}
        self.W_REF     = {"main":  float(os.environ.get("R3_WREF_MAIN", "8")),
                          "minor": float(os.environ.get("R3_WREF_MINOR", "4"))}
        # 計測: 各青で設定した MIN の合計と回数（平均MINを見る）
        self.min_sum = {"main": 0.0, "minor": 0.0}
        self.min_cnt = {"main": 0, "minor": 0}

    def _predict_W(self, road_id, node_index):
        w9x2 = self.tracker.model_input()
        if w9x2 is None:
            return None
        wait_9 = w9x2.sum(axis=1)
        n_cur = self.tracker.latest_N.get(road_id, 0)
        model, sw, si, node = self.dual[road_id]
        W_pred, _N = arr.predict_WN(wait_9, n_cur, model, sw, si, self.edge_index, node)
        return max(0.0, float(W_pred))

    def _min_from_W(self, key, W):
        base = self.BASE_MIN[key]; floor = self.MIN_FLOOR[key]; wref = self.W_REF[key]
        return int(round(floor + (base - floor) * min(1.0, W / wref)))

    def update(self, step):
        phase = traci.trafficlight.getPhase(self.tls_id)
        # 青開始時: W予測 → MIN_GREEN を需要に応じて設定
        if phase == self.MAIN_GREEN and not self.main_green_active:
            W = self._predict_W("1", 0)
            if W is not None:
                self.MIN_MAIN_GREEN = self._min_from_W("main", W)
                self.min_sum["main"] += self.MIN_MAIN_GREEN; self.min_cnt["main"] += 1
        if phase == self.MINOR_GREEN and not self.minor_green_active:
            W = self._predict_W("5", 4)
            if W is not None:
                self.MIN_MINOR_GREEN = self._min_from_W("minor", W)
                self.min_sum["minor"] += self.MIN_MINOR_GREEN; self.min_cnt["minor"] += 1
        # 骨格のギャップ制御（ギャップ閾値は MAX_GAP=5s のまま）
        super().update(step)

    def dump_stats(self, out_dir):
        path = os.path.join(out_dir, "予測ギャップMIN調整_ログ.csv")
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["対象", "基準MIN", "下限MIN", "W_REF", "青回数", "平均設定MIN(秒)"])
            for k, lbl in [("main", "主道路(道路1)"), ("minor", "従道路(道路5)")]:
                avg = round(self.min_sum[k] / self.min_cnt[k], 1) if self.min_cnt[k] else 0
                w.writerow([lbl, self.BASE_MIN[k], self.MIN_FLOOR[k], self.W_REF[k],
                            self.min_cnt[k], avg])
        print(f"✅ 予測ギャップMIN調整ログ: {path}")
        for k, lbl in [("main", "主"), ("minor", "従")]:
            avg = round(self.min_sum[k] / self.min_cnt[k], 1) if self.min_cnt[k] else 0
            print(f"   [{lbl}] 平均設定MIN={avg}s (基準{self.BASE_MIN[k]}s)")


# ============================================================
# グローバル状態
# ============================================================
last_phase_delta         = None   # 直前サイクルで適用した道路1のdelta
applied_delta_road1      = 0      # 直前サイクルで実際に適用したdelta（計測タイミング補正用）
signal_A_adjusted_once   = False
cycle_count_road1        = 0      # 道路1 制御サイクルカウンタ
cumulative_control_road1 = 0      # 累積制御偏差（道路1）: 60sベースからの偏差合計
road1_adjustment_pending = False  # 調整サイクル予約フラグ（道路1）

# ★ 道路5 用グローバル
last_phase_delta_road5   = None   # 直前サイクルで適用した道路5のdelta5
applied_delta_road5      = 0      # 直前サイクルで実際に適用したdelta5（計測タイミング補正用）
signal_road5_adjusted_once   = False
prediction_count_road5       = 0
cycle_count_road5            = 0   # 道路5 制御サイクルカウンタ
cumulative_control_road5 = 0      # 累積制御偏差（道路5）
road5_adjustment_pending = False  # 調整サイクル予約フラグ（道路5）
latest_data_road5 = {rid: None for rid in used_road_ids}

green_monitoring_active      = False
green_monitor_start_time     = None
green_monitor_end_time       = None
green_monitor_no_pass_seconds = 0
green_monitor_phase_end_time = None

main_green_durations  = []
minor_green_durations = []


# ============================================================
# シミュレーション本体
# ============================================================
def run_simulation(sumocfg_path, log_dir_path, control_signal=True):
    global prediction_count, last_phase_delta, signal_A_adjusted_once, cycle_count_road1, applied_delta_road1
    global prediction_count_road5, last_phase_delta_road5, signal_road5_adjusted_once, cycle_count_road5, applied_delta_road5
    global latest_data_road5
    global green_monitoring_active, green_monitor_start_time
    global green_monitor_end_time, green_monitor_no_pass_seconds
    global green_monitor_phase_end_time

    if not os.path.exists(log_dir_path):
        os.makedirs(log_dir_path)

    # === 赤2分割モデル＋スケーラー読み込み（既定: 分散評価あり / R2_MODEL1・R2_MODEL5 で変更可） ===
    model_road1, scaler_road1 = red2.load_road_model(MODEL_DIR_ROAD1)
    print(f"✅ 道路1 赤2分割モデル＋スケーラー読み込み完了（{MODEL_DIR_ROAD1}）")
    model_road5, scaler_road5 = red2.load_road_model(MODEL_DIR_ROAD5)
    print(f"✅ 道路5 赤2分割モデル＋スケーラー読み込み完了（{MODEL_DIR_ROAD5}）")

    # === 赤2分割の実時間計測トラッカー（全9道路・初停車ビン方式） ===
    red2_tracker = red2.RedSplit2Tracker()

    sumoBinary = sumolib.checkBinary('sumo-gui' if USE_GUI else 'sumo')
    traci.start([sumoBinary, "-c", sumocfg_path, "--start", "--quit-on-end"])
    sim_time = 0
    prev_states           = defaultdict(dict)
    pending_counts        = defaultdict(list)
    first_half_storage    = defaultdict(dict)

    writers = {}
    scheduled_times_A = set()
    last_phase_delta_before_prediction       = 0
    last_phase_delta_road5_before_prediction = 0   # ★ 道路5 用

    vehicle_delay_time = {}

    queue_delay_writers = {}
    vehicle_delay_time  = {}
    cycle_stopped_vehicles = {rid: set() for rid in ["1", "5", "6", "10"]}

    # === 交差点A 出力フォルダ ===
    a_dir = os.path.join(log_dir_path, "交差点A")
    os.makedirs(a_dir, exist_ok=True)

    total_file_path = os.path.join(a_dir, "交差点A_total_queue_delay.csv")
    total_file      = open(total_file_path, "w", newline="", encoding="utf-8-sig")
    total_writer    = csv.writer(total_file)
    total_writer.writerow(["サイクル", "step", "全道路停車車両数合計", "全道路遅れ時間合計"])

    queue_delay_files   = {}
    queue_delay_writers = {}

    for road_id in ["1", "5", "6", "10"]:
        file_path = os.path.join(a_dir, f"道路{road_id}_queue_delay.csv")
        f      = open(file_path, "w", newline="", encoding="utf-8-sig")
        writer = csv.writer(f)
        writer.writerow(["サイクル", "step", "道路ID", "待ち台数合計", "遅れ時間合計"])
        queue_delay_files[road_id]   = f
        queue_delay_writers[road_id] = writer

    cycle_queue_counts = {rid: [] for rid in ["1", "12", "10", "11"]}
    cycle_delay_sums   = {rid: [] for rid in ["1", "12", "10", "11"]}
    cycle_counter      = 0
    last_phase_A       = None
    _g1_start          = None   # 時間帯別 青計測: 道路1(phase0) 青開始step
    _g5_start          = None   # 時間帯別 青計測: 道路5(phase5) 青開始step

    # J交差点道路（1/5/6/10）の実測赤時間計測用
    j_road_red_start   = {}   # {road_id: 赤開始ステップ}
    j_road_red_history = {}   # {road_id: {step: 台数}} 赤期間の全ステップ記録
    # J交差点 設定マップ（毎ループで参照するため先引き）
    _J_CONF_MAP = {conf["id"]: conf["edges"] for conf in TRAFFIC_LIGHT_CONFIG["J"]}

    main_green_durations  = []
    minor_green_durations = []
    green_start_time      = None
    current_green_type    = None

    cycle_count = 0
    green_state = {"start": None, "type": None}

    tls_id = "A"
    MAIN_GREEN_PHASES  = [0]
    MINOR_GREEN_PHASES = [6]

    for tl_id, configs in TRAFFIC_LIGHT_CONFIG.items():
        for conf in configs:
            road_id = conf["id"]
            f = open(os.path.join(log_dir_path, f"道路{road_id}.csv"), "w", newline="", encoding="utf-8")
            writer = csv.writer(f)
            writer.writerow(["step", "count"])
            writers[road_id] = writer

    print("🚦 シミュレーション開始...")

    # ギャップ感応制御インスタンス（gap モード時のみ）
    if CONTROL_MODE == "gap":
        gap_controller = GapCycleController("J")
        print("🔀 ギャップ感応制御モード で起動")
    elif CONTROL_MODE == "pgap":
        gap_controller = PredictiveGapController("J", red2_tracker, edge_index)
        print("🔮 予測ギャップ感応制御モード で起動（ギャップ骨格+到着間隔N）")
    elif CONTROL_MODE == "pgapmin":
        gap_controller = MinGreenPredGapController("J", red2_tracker, edge_index)
        print("🔮 予測ギャップ【MIN調整版】で起動（W予測でMIN_GREEN可変・ギャップ5s維持）")
    elif CONTROL_MODE == "npred":
        gap_controller = PredNOnlyController("J", red2_tracker, edge_index)
        print("🎯 N単独制御モード で起動（ギャップ無視・通過台数≥予測Nで青を切る）")
    else:
        gap_controller = None
        if CONTROL_MODE == "prediction":
            print(f"🤖 GNN予測制御モード で起動（制御案={CONTROL_VARIANT}）")
        else:
            print("⛔ 制御なしモード で起動（固定サイクル）")

    # === 時間帯別サイクル長制御のセットアップ ===
    # 全120s信号に対し、各目標サイクル長のスケール済みプログラムを事前生成しておく。
    # J交差点はGNN制御でphase0/phase5を可変にするため、ここでは J 以外を対象にする。
    cycle_scaled_logics = {}   # tl_id -> {cycle_length: Logic}
    _target_cycles = {red2.PREP_CYCLE_LENGTH} | {c for _, _, c in red2.CYCLE_SCHEDULE_BY_HOUR}
    for tl_id in traci.trafficlight.getIDList():
        if tl_id == "J":
            continue   # J は GNN制御側で apply_j_logic により別途スケジュール
        try:
            base_logic = traci.trafficlight.getAllProgramLogics(tl_id)[0]
        except Exception:
            continue
        base_cycle = sum(ph.duration for ph in base_logic.phases)
        if abs(base_cycle - red2.BASE_CYCLE_LENGTH) > 0.5:
            continue   # 120s以外は対象外
        scaled = {}
        for cyc in _target_cycles:
            lg = red2.build_scaled_logic(traci, base_logic, cyc)
            if lg is not None:
                scaled[cyc] = lg
        if scaled:
            cycle_scaled_logics[tl_id] = scaled
    print(f"✅ サイクル長 時間帯別制御: 対象信号 {len(cycle_scaled_logics)} 個 / 目標長 {sorted(_target_cycles)}s")
    current_applied_cycle = None
    signal_stopped_prev = {}   # 前ステップに各信号で停車していた車両集合（待ち台数の新規判定用）

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        sim_time = traci.simulation.getTime()

        # === 終了判定（ループ先頭で確実に。R3_END/SIM_END_TIME 到達で打ち切り）===
        if sim_time >= SIM_END_TIME:
            print(f"🛑 SIM_END_TIME {SIM_END_TIME}s 到達 → 終了")
            break

        # === 赤2分割の実時間計測（全9道路・初停車ビン）: 毎ステップ更新 ===
        #   finalized: このステップで赤終了し2分割が確定した道路のリスト。
        #   ※ 旧「赤終了ブロック」は j_road_red_start 依存（green文字列がプレースホルダで
        #     機能しない）ため、予測・制御はトラッカーの確定シグナルから直接発火させる。
        finalized_roads = red2_tracker.step(traci, sim_time, PREP_TIME)

        if sim_time >= PREP_TIME and CONTROL_MODE == "prediction" and finalized_roads:
            _wait_9x2 = red2_tracker.model_input()
            if _wait_9x2 is not None:
                # --- 道路1（node0 / phase0）---
                if "1" in finalized_roads:
                    _pred1 = run_prediction_red2(_wait_9x2, 0, model_road1, scaler_road1, edge_index)
                    prediction_count += 1
                    prediction_writer.writerow([prediction_count, int(sim_time),
                                                round(float(_pred1[0]), 2), round(float(_pred1[1]), 2),
                                                round(float(np.sum(_pred1)), 2)])
                    print(f"🔮 [道路1][{prediction_count}] 予測[t3,t4]={np.round(_pred1, 2)} (step={int(sim_time)})")
                    if control_signal:
                        control_j_road(0, "1", _pred1, int(sim_time), prediction_count)
                # --- 道路5（node4 / phase5）---
                if "5" in finalized_roads:
                    _pred5 = run_prediction_red2(_wait_9x2, 4, model_road5, scaler_road5, edge_index)
                    prediction_count_road5 += 1
                    prediction_writer_road5.writerow([prediction_count_road5, int(sim_time),
                                                      round(float(_pred5[0]), 2), round(float(_pred5[1]), 2),
                                                      round(float(np.sum(_pred5)), 2)])
                    print(f"🔮 [道路5][{prediction_count_road5}] 予測[t3,t4]={np.round(_pred5, 2)} (step={int(sim_time)})")
                    if control_signal:
                        control_j_road(5, "5", _pred5, int(sim_time), prediction_count_road5)

        # === A〜P 全信号の 遅れ時間 & 待ち台数 計測: 全24h と 6〜19時窓 ===
        if sim_time >= PREP_TIME:
            _inwin = in_analysis_window(sim_time)
            _hour  = int((sim_time - PREP_TIME) // 3600)   # 時間帯（step_to_hourと同基準）
            _cur_stopped = defaultdict(set)
            for _veh in traci.vehicle.getIDList():
                if traci.vehicle.getSpeed(_veh) <= 0:
                    _sig = edge_to_signal.get(traci.vehicle.getRoadID(_veh))
                    if _sig is not None:
                        signal_delay_full[_sig] += 1          # 遅れ=停車車両×秒
                        if _inwin:
                            signal_delay_win[_sig] += 1
                        _cur_stopped[_sig].add(_veh)
                        delay_hour_all[_hour] += 1            # 時間帯別（街全体）
                        if _sig == TARGET_SIGNAL:
                            delay_hour_J[_hour] += 1          # 時間帯別（J）
            # 待ち台数=延べ停車台数: 新規に停車した車だけ計上（前ステップに無かった車）
            for _sig, _s in _cur_stopped.items():
                _new = len(_s - signal_stopped_prev.get(_sig, _EMPTY_SET))
                if _new:
                    signal_queue_full[_sig] += _new
                    if _inwin:
                        signal_queue_win[_sig] += _new
                    queue_hour_all[_hour] += _new             # 時間帯別（街全体）
                    if _sig == TARGET_SIGNAL:
                        queue_hour_J[_hour] += _new           # 時間帯別（J）
            signal_stopped_prev = _cur_stopped

        # === 時間帯別サイクル長の切替（目標が変わった時だけ全対象信号を差し替え）===
        _target_cycle = red2.get_target_cycle(sim_time, PREP_TIME)
        if _target_cycle != current_applied_cycle:
            for tl_id, scaled in cycle_scaled_logics.items():
                lg = scaled.get(_target_cycle)
                if lg is None:
                    continue
                try:
                    lg.currentPhaseIndex = traci.trafficlight.getPhase(tl_id)
                    traci.trafficlight.setProgramLogic(tl_id, lg)
                except Exception as e:
                    print(f"⚠️ サイクル長切替失敗 {tl_id}: {e}")
            # J交差点: スケジュール青は全モードで更新する（時間帯別サイクル長に追従させるため）。
            #   ・予測 / 制御なし … apply_j_logic() で位相長に反映する。
            #   ・gap / pgap 系   … 位相長はギャップ制御が占有して設定するので apply_j_logic は
            #                       呼ばず、代わりに可動範囲（スケジュール青±10s）を取り直す。
            #     ※ 以前は gap 系でスケジュール青の更新自体を飛ばしていたため、
            #        他の信号(A〜P)が100/110/130sに切り替わっても J だけ 120s 基準の
            #        固定青(60/26s)のままで、サイクル長が食い違っていた。
            update_j_scheduled_green(_target_cycle)
            if CONTROL_MODE not in ("gap", "pgap", "pgapmin", "npred"):
                apply_j_logic()
            elif gap_controller is not None:
                gap_controller._refresh_bounds()
                gap_controller._apply_max_durations()
            print(f"🔄 step={sim_time:.0f}s: 目標サイクル長を {_target_cycle}s に変更")
            current_applied_cycle = _target_cycle

        # ギャップ感応制御（gap モード時のみ毎ステップ呼び出し）
        if gap_controller is not None:
            gap_controller.update(sim_time)

        phase = traci.trafficlight.getPhase("J")

        if phase in MAIN_GREEN_PHASES or phase in MINOR_GREEN_PHASES:
            if green_state["start"] is None:
                green_state["start"] = sim_time
                green_state["type"]  = "主道路" if phase in MAIN_GREEN_PHASES else "従道路"
        else:
            if green_state["start"] is not None:
                duration = sim_time - green_state["start"]
                cycle_count += 1
                green_writer.writerow([cycle_count, sim_time, green_state["type"], duration])
                print(f"🟢 {green_state['type']} 青 {duration} 秒 (step={sim_time})")
                green_state["start"] = None
                green_state["type"]  = None

        # === サイクル判定 ===
        current_phase_A = traci.trafficlight.getPhase("J")
        total_phases_A  = len(traci.trafficlight.getAllProgramLogics("J")[0].getPhases())

        # === 時間帯別: 青時間(道路1=phase0 / 道路5=phase5) の実測 ===
        #   フェーズ遷移で青の開始・終了を検出し、開始時刻の属する時間帯へ加算。
        if sim_time >= PREP_TIME and last_phase_A is not None:
            if current_phase_A == 0 and last_phase_A != 0:
                _g1_start = sim_time
            elif current_phase_A != 0 and last_phase_A == 0 and _g1_start is not None:
                _hh = int((_g1_start - PREP_TIME) // 3600)
                green_hour_sum["主道路"][_hh] += (sim_time - _g1_start)
                green_hour_cnt["主道路"][_hh] += 1
                _g1_start = None
            if current_phase_A == 5 and last_phase_A != 5:
                _g5_start = sim_time
            elif current_phase_A != 5 and last_phase_A == 5 and _g5_start is not None:
                _hh = int((_g5_start - PREP_TIME) // 3600)
                green_hour_sum["従道路"][_hh] += (sim_time - _g5_start)
                green_hour_cnt["従道路"][_hh] += 1
                _g5_start = None

        if last_phase_A == total_phases_A - 1 and current_phase_A == 0 and sim_time >= 1200:
            cycle_counter += 1
            if sim_time >= PREP_TIME:
                cycle_hour_cnt[int((sim_time - PREP_TIME) // 3600)] += 1
            total_stopped = 0
            total_delay   = 0

            for road_id in ["1", "5", "6", "10"]:
                stopped_count = len(cycle_stopped_vehicles[road_id])
                delay_sum     = sum(vehicle_delay_time.get(veh, 0) for veh in cycle_stopped_vehicles[road_id])

                queue_delay_writers[road_id].writerow(
                    [cycle_counter, sim_time, road_id, stopped_count, delay_sum]
                )
                total_stopped += stopped_count
                total_delay   += delay_sum

            total_writer.writerow([cycle_counter, sim_time, total_stopped, total_delay])
            cycle_stopped_vehicles = {rid: set() for rid in ["1", "5", "6", "10"]}

        # ============================================================
        # J交差点 道路別 赤終了検知（フェーズ遷移ベース / 全制御モード対応）
        # 主道路(1/10): フェーズ→0 で赤終了
        # 従道路(5/6) : フェーズ→6 で赤終了
        # ============================================================
        if sim_time >= PREP_TIME and last_phase_A is not None:

            # ── 主道路（1/10）赤終了: フェーズが 0 になった瞬間 ──────────────
            if last_phase_A != 0 and current_phase_A == 0:
                for rid in ["1", "10"]:
                    if rid not in j_road_red_start:
                        continue
                    t0         = j_road_red_start.pop(rid)
                    hist       = j_road_red_history.pop(rid, {})
                    actual_red = int(sim_time) - t0
                    t1_step    = t0 + max(1, round(actual_red * 0.50))   # 赤時間50%（中間点）
                    t2_step    = t0 + actual_red - 1                       # 赤時間100%（最終ステップ）
                    cnt_t1 = hist.get(t1_step) or (hist.get(min(hist, key=lambda k: abs(k - t1_step), default=t0), 0) if hist else 0)
                    cnt_t2 = hist.get(t2_step) or (hist.get(min(hist, key=lambda k: abs(k - t2_step), default=t0), 0) if hist else 0)
                    diff       = max(0, cnt_t2 - cnt_t1)
                    delay_sec  = cnt_t1 * (actual_red * 0.25) + diff * (actual_red * 0.75)
                    print(f"📐 [赤終了/{rid}] step={int(sim_time)}, 赤={actual_red}s, 前半(50%)={cnt_t1}, 後半増分(100%)={diff}, 遅れ={int(round(delay_sec))}s")

                    writers[rid].writerow([int(sim_time), cnt_t1])
                    writers[rid].writerow([int(sim_time), diff])
                    latest_data[rid]       = (cnt_t1, diff)
                    latest_data_road5[rid] = (cnt_t1, diff)
                    logger.log_measurement_data(rid, cnt_t1, diff, int(sim_time))

                    if rid == "1":
                        delay_writer.writerow([prediction_count, int(sim_time), cnt_t1, diff, 0, actual_red, int(round(delay_sec))])

                        # === 赤2分割: 全9道路の最新2分割 → 道路1(node0)の [t3,t4] を予測 ===
                        wait_9x2 = red2_tracker.model_input()
                        if wait_9x2 is not None:
                            pred = run_prediction_red2(wait_9x2, 0, model_road1, scaler_road1, edge_index)
                            prediction_count += 1
                            print(f"🔮 [道路1][{prediction_count}] 予測[t3,t4]={np.round(pred, 2)} (step={int(sim_time)})")
                            if sim_time >= SIM_END_TIME:
                                print(f"🛑 終了時刻 {SIM_END_TIME}s に到達。シミュレーション終了。")
                                traci.close()
                                _close_all_logs(queue_delay_files, total_file)
                                return
                            prediction_writer.writerow([prediction_count, int(sim_time),
                                                        round(float(pred[0]), 2), round(float(pred[1]), 2),
                                                        round(float(np.sum(pred)), 2)])
                            if control_signal and CONTROL_MODE == "prediction":
                                control_j_road(0, "1", pred, int(sim_time), prediction_count)
                        else:
                            logger.log_data_incomplete(int(sim_time), label="道路1 ")

                    elif rid == "10":
                        noth_delay_writer.writerow([prediction_count, int(sim_time), cnt_t1, diff, 0, actual_red, int(round(delay_sec))])

            # ── 従道路（5/6）赤終了: フェーズが 6 になった瞬間 ──────────────
            elif last_phase_A != 6 and current_phase_A == 6:
                for rid in ["5", "6"]:
                    if rid not in j_road_red_start:
                        continue
                    t0         = j_road_red_start.pop(rid)
                    hist       = j_road_red_history.pop(rid, {})
                    actual_red = int(sim_time) - t0
                    t1_step    = t0 + max(1, round(actual_red * 0.50))   # 赤時間50%（中間点）
                    t2_step    = t0 + actual_red - 1                       # 赤時間100%（最終ステップ）
                    cnt_t1 = hist.get(t1_step) or (hist.get(min(hist, key=lambda k: abs(k - t1_step), default=t0), 0) if hist else 0)
                    cnt_t2 = hist.get(t2_step) or (hist.get(min(hist, key=lambda k: abs(k - t2_step), default=t0), 0) if hist else 0)
                    diff       = max(0, cnt_t2 - cnt_t1)
                    delay_sec  = cnt_t1 * (actual_red * 0.25) + diff * (actual_red * 0.75)
                    print(f"📐 [赤終了/{rid}] step={int(sim_time)}, 赤={actual_red}s, 前半(50%)={cnt_t1}, 後半増分(100%)={diff}, 遅れ={int(round(delay_sec))}s")

                    writers[rid].writerow([int(sim_time), cnt_t1])
                    writers[rid].writerow([int(sim_time), diff])
                    latest_data[rid]       = (cnt_t1, diff)
                    latest_data_road5[rid] = (cnt_t1, diff)
                    logger.log_measurement_data(rid, cnt_t1, diff, int(sim_time))

                    if rid == "5":
                        delay_writer_road5.writerow([prediction_count_road5, int(sim_time), cnt_t1, diff, 0, actual_red, int(round(delay_sec))])

                        # === 赤2分割: 全9道路の最新2分割 → 道路5(node4)の [t3,t4] を予測 ===
                        wait_9x2 = red2_tracker.model_input()
                        if wait_9x2 is not None:
                            pred5 = run_prediction_red2(wait_9x2, 4, model_road5, scaler_road5, edge_index)
                            prediction_count_road5 += 1
                            print(f"🔮 [道路5][{prediction_count_road5}] 予測[t3,t4]={np.round(pred5, 2)} (step={int(sim_time)})")
                            prediction_writer_road5.writerow([prediction_count_road5, int(sim_time),
                                                              round(float(pred5[0]), 2), round(float(pred5[1]), 2),
                                                              round(float(np.sum(pred5)), 2)])
                            if control_signal and CONTROL_MODE == "prediction":
                                # 道路5の青 = phase5（現ネット。旧phase6は旧ネットの名残）
                                control_j_road(5, "5", pred5, int(sim_time), prediction_count_road5)
                        else:
                            logger.log_data_incomplete(int(sim_time), label="道路5 ")

                    elif rid == "6":
                        delay_minor_writer.writerow([prediction_count, int(sim_time), rid, cnt_t1, diff, 0, actual_red, int(round(delay_sec))])

        last_phase_A = current_phase_A

        # === J交差点 赤期間中の全ステップ台数記録 ===
        # RED END 検知ブロックの後に実行することで、赤終了ステップ（= 青開始ステップ）を
        # 履歴に含めないようにしている（pop済みの road_id は keys() に現れない）
        if sim_time >= PREP_TIME:
            for rid in list(j_road_red_start.keys()):
                c = 0
                for e in _J_CONF_MAP.get(rid, []):
                    for v in traci.edge.getLastStepVehicleIDs(e):
                        if traci.vehicle.getSpeed(v) <= 0:
                            c += 1
                j_road_red_history.setdefault(rid, {})[int(sim_time)] = math.ceil(c / 2)

        # === 時間距離図 ===
        log_vehicle_positions(sim_time)

        # === 待ち台数・遅れ時間の記録 ===
        if sim_time >= 1200:
            for road_id in ["1", "5", "6", "10"]:
                edges = None
                for tl_id, configs in TRAFFIC_LIGHT_CONFIG.items():
                    for conf in configs:
                        if conf["id"] == road_id:
                            edges = conf["edges"]
                            break
                    if edges:
                        break
                if not edges:
                    continue

                for edge in edges:
                    for veh_id in traci.edge.getLastStepVehicleIDs(edge):
                        if traci.vehicle.getSpeed(veh_id) <= 0:
                            cycle_stopped_vehicles[road_id].add(veh_id)
                            if veh_id not in vehicle_delay_time:
                                vehicle_delay_time[veh_id] = 0
                            vehicle_delay_time[veh_id] += 1

        # === 無駄青監視 ===
        if green_monitoring_active and green_monitor_start_time <= sim_time < green_monitor_end_time:
            try:
                passing = traci.edge.getLastStepVehicleIDs("E24")
                if len(passing) == 0:
                    green_monitor_no_pass_seconds += 1
            except Exception as e:
                print(f"⚠️ 無駄青監視エラー: {e}")

        elif green_monitoring_active and sim_time >= green_monitor_end_time:
            monitoring_duration = green_monitor_end_time - green_monitor_start_time
            if sim_time >= SIM_END_TIME:
                print(f"🛑 終了時刻 {SIM_END_TIME}s に到達。シミュレーション終了。")
                traci.close()
                _close_all_logs()
                return

            wasted_green_writer.writerow([
                prediction_count,
                green_monitor_phase_end_time,
                green_monitor_end_time - green_monitor_start_time,
                green_monitor_no_pass_seconds
            ])
            green_monitoring_active = False

        # ============================================================
        # 信号状態監視・計測スケジュール
        # ============================================================
        for tl_id, configs in TRAFFIC_LIGHT_CONFIG.items():
            for i, config in enumerate(configs):
                try:
                    key     = (tl_id, i)
                    current = traci.trafficlight.getRedYellowGreenState(tl_id)
                    prev    = prev_states[tl_id].get(i, "")

                    if sim_time >= PREP_TIME and prev == config["red"] and current == config["green"]:
                        road_id = config["id"]
                        edges   = config["edges"]

                        # 無駄青監視
                        if config["edges"] == ["DtoA"]:
                            logic            = traci.trafficlight.getAllProgramLogics("J")[0]
                            phase_0_duration = logic.getPhases()[0].duration
                            green_monitor_start_time     = int(sim_time)
                            green_monitor_end_time       = int(sim_time + phase_0_duration)
                            green_monitor_phase_end_time = green_monitor_end_time
                            green_monitor_no_pass_seconds = 0
                            green_monitoring_active      = True

                        # --- 測定タイミング設定 ---
                        if road_id in ["1", "10", "5", "6"]:
                            # 赤開始：履歴辞書を初期化し、初期台数を記録
                            cnt = 0
                            for e in edges:
                                for v in traci.edge.getLastStepVehicleIDs(e):
                                    if traci.vehicle.getSpeed(v) <= 0:
                                        cnt += 1
                            cnt = math.ceil(cnt / 2)
                            j_road_red_start[road_id]               = int(sim_time)
                            j_road_red_history[road_id]             = {}
                            j_road_red_history[road_id][int(sim_time)] = cnt
                            print(f"🔴 [赤開始/{road_id}] step={int(sim_time)}, 停車={cnt}")

                        elif road_id in ["11", "14"]:
                            t1 = int(sim_time + 27); t2 = int(sim_time + 54)
                            pending_counts[key].append((t1, "t1", road_id, edges, 27))
                            pending_counts[key].append((t2, "t2", road_id, edges, 27))

                        elif road_id in ["12", "13"]:
                            t1 = int(sim_time + 44); t2 = int(sim_time + 88)
                            pending_counts[key].append((t1, "t1", road_id, edges, 44))
                            pending_counts[key].append((t2, "t2", road_id, edges, 44))

                        elif road_id in ["15", "18", "19", "22", "24", "25", "27", "29", "30",
                                         "32", "35", "36", "39", "40", "42", "43", "45", "48",
                                         "49", "50", "52", "53", "56", "57", "60", "61"]:
                            t1 = int(sim_time + 42); t2 = int(sim_time + 83)
                            pending_counts[key].append((t1, "t1", road_id, edges, 41))
                            pending_counts[key].append((t2, "t2", road_id, edges, 41))

                        else:
                            t1 = int(sim_time + 21); t2 = int(sim_time + 41)
                            pending_counts[key].append((t1, "t1", road_id, edges, 20))
                            pending_counts[key].append((t2, "t2", road_id, edges, 20))

                    # --- 計測実行（道路1/5/6/10は赤終了フェーズ検知で処理済み）---
                    new_pending = []
                    for record in pending_counts[key]:
                        record_time, tag, road_id, edges, expected_diff = record[:5]
                        if sim_time >= record_time:
                            count = 0
                            for edge in edges:
                                for veh_id in traci.edge.getLastStepVehicleIDs(edge):
                                    if traci.vehicle.getSpeed(veh_id) <= 0:
                                        count += 1

                            try:
                                id_int = int(road_id)
                                if id_int in [1, 2, 3, 4, 9, 12, 23, 25, 28, 40, 47, 50, 51, 54, 55, 58, 59, 62]:
                                    count = math.ceil(count / 2)
                                elif id_int in [5, 6, 7, 8, 10, 11, 14, 15, 18, 19, 30, 31, 34, 35, 37, 38, 42, 44, 45]:
                                    count = math.ceil(count / 2)
                            except:
                                pass

                            if tag == "t1":
                                first_half_storage[road_id][record_time] = (count, expected_diff)
                                writers[road_id].writerow([record_time, count])

                            elif tag == "t2":
                                matched = False
                                for t1_time, (prev_count, diff_exp) in list(first_half_storage[road_id].items()):
                                    if abs(record_time - t1_time - diff_exp) <= 1:
                                        diff = count - prev_count

                                        if diff < 0:
                                            print(f"⚠️ 計測異常: 道路{road_id} の差分が負です。t1={prev_count}, t2={count} → 差={diff} (step={record_time})")

                                        # --- 遅れ時間ログ（既存） ---
                                        if road_id == "6":
                                            # 従道路の赤時間はフェーズ0（主道路青）の延長分だけ長くなる
                                            # → 道路1制御による delta1 を使用
                                            delta_for_log = last_phase_delta_before_prediction if (control_signal and signal_A_adjusted_once) else 0
                                            okure = 91 + delta_for_log
                                            delay_seconds = prev_count * (okure * 0.75) + diff * (okure * 0.25)
                                            delay_minor_writer.writerow([
                                                prediction_count, record_time, road_id,
                                                prev_count, diff, delta_for_log, okure, int(round(delay_seconds))
                                            ])

                                        if road_id == "10":
                                            # 主道路の赤時間はフェーズ6（従道路青）の延長分だけ長くなる
                                            # → 道路5制御による delta5 を使用
                                            delta_for_log = last_phase_delta_road5_before_prediction if (control_signal and signal_road5_adjusted_once) else 0
                                            okure = 57 - delta_for_log
                                            delay_seconds = prev_count * (okure * 0.75) + diff * (okure * 0.25)
                                            noth_delay_writer.writerow([
                                                prediction_count, record_time,
                                                prev_count, diff, delta_for_log, okure, int(round(delay_seconds))
                                            ])

                                        if road_id in ["11", "14"]:
                                            okure = 57
                                            delay_seconds = prev_count * (okure * 0.75) + diff * (okure * 0.25)
                                            all_delay_writer.writerow([
                                                prediction_count, record_time, road_id,
                                                prev_count, diff, okure, int(round(delay_seconds))
                                            ])

                                        if road_id in ["12", "13"]:
                                            okure = 91
                                            delay_seconds = prev_count * (okure * 0.75) + diff * (okure * 0.25)
                                            all_delay_writer.writerow([
                                                prediction_count, record_time, road_id,
                                                prev_count, diff, okure, int(round(delay_seconds))
                                            ])

                                        if road_id in ["11", "14", "16", "17", "20", "21", "23", "26", "2", "28",
                                                        "3", "31", "33", "34", "37", "38", "41", "4", "44", "7",
                                                        "46", "47", "8", "51", "9", "54", "55", "58", "59", "62"]:
                                            okure = 42
                                            delay_seconds = prev_count * (okure * 0.75) + diff * (okure * 0.25)
                                            all_delay_writer.writerow([
                                                prediction_count, record_time, road_id,
                                                prev_count, diff, okure, int(round(delay_seconds))
                                            ])

                                        if road_id in ["15", "18", "19", "22", "24", "25", "27", "29", "30",
                                                        "32", "35", "36", "39", "40", "42", "43", "45", "48",
                                                        "49", "50", "52", "53", "56", "57", "60", "61"]:
                                            okure = 84
                                            delay_seconds = prev_count * (okure * 0.75) + diff * (okure * 0.25)
                                            all_delay_minor_writer.writerow([
                                                prediction_count, record_time, road_id,
                                                prev_count, diff, okure, int(round(delay_seconds))
                                            ])

                                        writers[road_id].writerow([record_time, diff])
                                        del first_half_storage[road_id][t1_time]
                                        latest_data[road_id] = (prev_count, diff)

                                        # ★ 道路5 予測用にもデータを蓄積
                                        latest_data_road5[road_id] = (prev_count, diff)

                                        logger.log_measurement_data(road_id, prev_count, diff, record_time)

                                        # ==========================================
                                        # 道路1 予測・制御（無効化）
                                        #   ※ 道路1は上の「動的赤終了検知」ブロックで
                                        #     赤2分割トラッカーを使って予測・制御する。
                                        #     ここ(pending_counts経路)には道路1は来ないため無効化。
                                        # ==========================================
                                        if False and road_id == "1":
                                            last_phase_delta_before_prediction = applied_delta_road1 if signal_A_adjusted_once else 0

                                            if all(latest_data[rid] is not None for rid in used_road_ids):
                                                logger.log_prediction_trigger(latest_data, label="[道路1] ")
                                                t1s = [latest_data[rid][0] for rid in used_road_ids]
                                                dts = [latest_data[rid][1] for rid in used_road_ids]

                                                pred = run_prediction(
                                                    t1_array=t1s,
                                                    delay_array=dts,
                                                    edge_index=edge_index,
                                                    scaler=scaler_road1,
                                                    model_path=model_path_road1,
                                                    prediction_count=prediction_count
                                                )

                                                pred_t3, pred_t4 = map(int, np.round(pred))
                                                t1_count = latest_data["1"][0]
                                                diff_val = latest_data["1"][1]

                                                prediction_count += 1
                                                logger.log_prediction_result(
                                                    prediction_count, t1_count, diff_val,
                                                    pred_t3, pred_t4, record_time, label="道路1"
                                                )

                                                if sim_time >= SIM_END_TIME:
                                                    print(f"🛑 終了時刻 {SIM_END_TIME}s に到達。シミュレーション終了。")
                                                    traci.close()
                                                    _close_all_logs(queue_delay_files, total_file)
                                                    return

                                                prediction_writer.writerow([
                                                    prediction_count, record_time, t1_count, diff_val, pred_t3, pred_t4
                                                ])

                                                # 道路1の赤時間はフェーズ6（従道路青）の延長分だけ長くなる
                                                # → 道路5制御による delta5 を使用
                                                delta_for_log = last_phase_delta_road5_before_prediction if (control_signal and signal_road5_adjusted_once) else 0
                                                okure = 57 + delta_for_log
                                                delay_seconds = t1_count * (okure * 0.75) + diff_val * (okure * 0.25)
                                                delay_writer.writerow([
                                                    prediction_count, record_time, t1_count, diff_val,
                                                    delta_for_log,
                                                    okure, int(round(delay_seconds))
                                                ])

                                                if control_signal and CONTROL_MODE == "prediction":
                                                    adjust_signal_A_based_on_prediction(
                                                        pred_t3, pred_t4, record_time, prediction_count,
                                                        pending_counts
                                                    )

                                                # latest_data["1"] を次サイクル予測用に更新
                                                t3       = pred_t3
                                                t4_actual    = count
                                                t4_minus_t3  = t4_actual - t3
                                                latest_data["1"] = (t3, t4_minus_t3)

                                            else:
                                                logger.log_data_incomplete(record_time, label="道路1 ")
                                                latest_data["1"] = None

                                        # ==========================================
                                        # ★ 道路5 予測・制御（無効化）
                                        #   ※ 道路5も上の「動的赤終了検知」ブロックで制御。
                                        #     ここ(pending_counts経路)には道路5は来ないため無効化。
                                        # ==========================================
                                        if False and road_id == "5":
                                            last_phase_delta_road5_before_prediction = (
                                                applied_delta_road5 if signal_road5_adjusted_once else 0
                                            )

                                            if all(latest_data_road5[rid] is not None for rid in used_road_ids):
                                                logger.log_prediction_trigger(latest_data_road5, label="[道路5] ")
                                                t1s_5 = [latest_data_road5[rid][0] for rid in used_road_ids]
                                                dts_5 = [latest_data_road5[rid][1] for rid in used_road_ids]

                                                pred5 = run_prediction(
                                                    t1_array=t1s_5,
                                                    delay_array=dts_5,
                                                    edge_index=edge_index,
                                                    scaler=scaler_road5,
                                                    model_path=model_path_road5,
                                                    prediction_count=prediction_count_road5
                                                )

                                                pred_t3_5, pred_t4_5 = map(int, np.round(pred5))
                                                t1_count_5 = latest_data_road5["5"][0]
                                                diff_val_5 = latest_data_road5["5"][1]

                                                prediction_count_road5 += 1
                                                logger.log_prediction_result(
                                                    prediction_count_road5, t1_count_5, diff_val_5,
                                                    pred_t3_5, pred_t4_5, record_time, label="道路5"
                                                )

                                                prediction_writer_road5.writerow([
                                                    prediction_count_road5, record_time,
                                                    t1_count_5, diff_val_5, pred_t3_5, pred_t4_5
                                                ])

                                                # 道路5 遅れ時間ログ
                                                # 道路5の赤時間はフェーズ0（主道路青）の延長分だけ長くなる
                                                # → 道路1制御による delta1 を使用
                                                delta_for_log5 = last_phase_delta_before_prediction if (control_signal and signal_A_adjusted_once) else 0
                                                okure5 = 91 + delta_for_log5
                                                delay_seconds5 = (
                                                    t1_count_5 * (okure5 * 0.75) +
                                                    diff_val_5  * (okure5 * 0.25)
                                                )
                                                delay_writer_road5.writerow([
                                                    prediction_count_road5, record_time,
                                                    t1_count_5, diff_val_5,
                                                    delta_for_log5, okure5, int(round(delay_seconds5))
                                                ])

                                                if control_signal and CONTROL_MODE == "prediction":
                                                    adjust_signal_based_on_prediction_road5(
                                                        pred_t3_5, pred_t4_5,
                                                        record_time, prediction_count_road5,
                                                        pending_counts
                                                    )

                                                # latest_data_road5["5"] を次サイクル用に更新
                                                t3_5        = pred_t3_5
                                                t4_actual_5 = count
                                                t4_minus_5  = t4_actual_5 - t3_5
                                                latest_data_road5["5"] = (t3_5, t4_minus_5)

                                            else:
                                                logger.log_data_incomplete(record_time, label="道路5 ")
                                                latest_data_road5["5"] = None

                                        matched = True
                                        break

                                if not matched:
                                    new_pending.append(record)
                        else:
                            new_pending.append(record)

                    pending_counts[key] = new_pending
                    prev_states[tl_id][i] = current

                except Exception as e:
                    print(f"⚠ Error at {tl_id} ({config['id']}) - {e}")

    if isinstance(gap_controller, (PredictiveGapController, MinGreenPredGapController)):
        gap_controller.dump_stats(OUT_DIR)
    traci.close()
    print("✅ シミュレーション完了")
    _close_all_logs(queue_delay_files, total_file)


def _close_all_logs(queue_delay_files=None, total_file=None):
    """全ログファイルを安全に閉じる。"""
    for obj in [
        phase_log_file, phase_log_file_road5,
        wasted_green_file,
        delay_log_file, delay_log_file_road5,
        noth_delay_log_file, delay_minor_log_file,
        all_delay_log_file, all_delay_minor_log_file,
        green_log_file, queue_delay_file,
        prediction_log_file, prediction_log_file_road5,
    ]:
        try:
            obj.close()
        except Exception:
            pass
    if queue_delay_files:
        for f in queue_delay_files.values():
            try:
                f.close()
            except Exception:
                pass
    if total_file:
        try:
            total_file.close()
        except Exception:
            pass


# ============================================================
# ポスト処理・グラフ描画（元のまま）
# ============================================================
import os
import shutil

def postprocess_logs(log_dir_path):
    signal_to_roads = defaultdict(list)
    for signal, configs in TRAFFIC_LIGHT_CONFIG.items():
        for conf in configs:
            signal_to_roads[signal].append(conf["id"])

    total_counts_by_signal = {}

    for signal, road_ids in signal_to_roads.items():
        signal_dir = os.path.join(log_dir_path, f"信号{signal}")
        os.makedirs(signal_dir, exist_ok=True)
        combined_df = None

        for road_id in road_ids:
            src_path = os.path.join(log_dir_path, f"道路{road_id}.csv")
            dst_path = os.path.join(signal_dir, f"道路{road_id}.csv")
            if not os.path.exists(src_path):
                continue
            shutil.move(src_path, dst_path)
            df = pd.read_csv(dst_path, encoding="utf-8-sig")
            df = df.rename(columns={"count": f"道路{road_id}"})
            if combined_df is None:
                combined_df = df
            else:
                combined_df = pd.merge(combined_df, df, on="step", how="outer")

        if combined_df is not None:
            combined_df = combined_df.sort_values("step").fillna(0)
            combined_df["合計"] = combined_df.drop(columns="step").sum(axis=1)
            signal_sum = int(combined_df["合計"].sum())
            combined_df.to_csv(os.path.join(signal_dir, f"信号{signal}_合計.csv"), index=False, encoding="utf-8-sig")
            total_counts_by_signal[signal] = signal_sum
        else:
            total_counts_by_signal[signal] = 0

    all_total = sum(total_counts_by_signal.values())
    total_df  = pd.DataFrame([total_counts_by_signal])
    total_df["全体合計"] = all_total
    total_df.to_csv(os.path.join(log_dir_path, "全体合計.csv"), index=False, encoding="utf-8-sig")
    print("✅ ポスト処理完了：信号機別フォルダ生成と全体合計出力")


def postprocess_intersection_summary():
    """
    全交差点の待ち台数・遅れ時間を交差点別に集計し、一つのCSVに出力する。
    出力列: 交差点, 前半台数合計, 後半台数合計, 遅れ時間合計(秒)
    末尾にJ交差点合計 / J以外合計 / 全体合計 の行を追加する。
    """
    output_path = os.path.join(OUT_DIR, "交差点別_待ち台数_遅れ時間_集計.csv")

    road_to_intersection = {}
    for tl_id, configs in TRAFFIC_LIGHT_CONFIG.items():
        for conf in configs:
            road_to_intersection[conf["id"]] = tl_id

    totals = {tl: {"前半台数合計": 0, "後半台数合計": 0, "遅れ時間合計": 0}
              for tl in TRAFFIC_LIGHT_CONFIG}

    def _add(tl, t1, t2_diff, delay):
        totals[tl]["前半台数合計"] += int(t1)
        totals[tl]["後半台数合計"] += int(t2_diff)
        totals[tl]["遅れ時間合計"]  += int(delay)

    # J交差点 道路1 / 道路5 / 道路10（専用ログファイル）
    for path in [delay_log_path, delay_log_path_road5, noth_delay_log_path]:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            df = pd.read_csv(path, encoding="utf-8-sig").fillna(0)
            for _, r in df.iterrows():
                _add("J", r["前半台数"], r["後半台数"], r["遅れ時間(秒)"])

    # J交差点 道路6（従道路ログから道路ID=6 のみ抽出）
    if os.path.exists(delay_minor_log_path) and os.path.getsize(delay_minor_log_path) > 0:
        df = pd.read_csv(delay_minor_log_path, encoding="utf-8-sig").fillna(0)
        for _, r in df[df["道路ID"].astype(str).str.strip() == "6"].iterrows():
            _add("J", r["前半台数"], r["後半台数"], r["遅れ時間(秒)"])

    # J以外 主道路（全遅れ時間ログ_主道路.csv）
    # 道路11・14 は重複して書かれているため (回数, 道路ID) で重複除去して最初の行を使用
    if os.path.exists(all_delay_log_path) and os.path.getsize(all_delay_log_path) > 0:
        df = pd.read_csv(all_delay_log_path, encoding="utf-8-sig").fillna(0)
        df["道路ID"] = df["道路ID"].apply(lambda x: str(int(float(x))))
        df = df.drop_duplicates(subset=["回数", "道路ID"], keep="first")
        for _, r in df.iterrows():
            tl = road_to_intersection.get(r["道路ID"])
            if tl and tl != "J":
                _add(tl, r["前半台数"], r["後半台数"], r["遅れ時間(秒)"])

    # J以外 従道路（全遅れ時間_従道路.csv）
    if os.path.exists(all_delay_minor_log_path) and os.path.getsize(all_delay_minor_log_path) > 0:
        df = pd.read_csv(all_delay_minor_log_path, encoding="utf-8-sig").fillna(0)
        df["道路ID"] = df["道路ID"].apply(lambda x: str(int(float(x))))
        df = df.drop_duplicates(subset=["回数", "道路ID"], keep="first")
        for _, r in df.iterrows():
            tl = road_to_intersection.get(r["道路ID"])
            if tl and tl != "J":
                _add(tl, r["前半台数"], r["後半台数"], r["遅れ時間(秒)"])

    # 集計表の作成
    intersection_order = list(TRAFFIC_LIGHT_CONFIG.keys())
    rows = []
    j_t1 = j_t2 = j_delay = 0
    non_j_t1 = non_j_t2 = non_j_delay = 0

    for tl in intersection_order:
        d = totals[tl]
        rows.append({
            "交差点":         f"交差点{tl}",
            "前半台数合計":   d["前半台数合計"],
            "後半台数合計":   d["後半台数合計"],
            "遅れ時間合計(秒)": d["遅れ時間合計"],
        })
        if tl == "J":
            j_t1, j_t2, j_delay = d["前半台数合計"], d["後半台数合計"], d["遅れ時間合計"]
        else:
            non_j_t1  += d["前半台数合計"]
            non_j_t2  += d["後半台数合計"]
            non_j_delay += d["遅れ時間合計"]

    # 区切り行 + サマリー行
    rows.append({"交差点": "", "前半台数合計": "", "後半台数合計": "", "遅れ時間合計(秒)": ""})
    rows.append({"交差点": "J交差点合計",  "前半台数合計": j_t1,            "後半台数合計": j_t2,            "遅れ時間合計(秒)": j_delay})
    rows.append({"交差点": "J以外合計",     "前半台数合計": non_j_t1,        "後半台数合計": non_j_t2,        "遅れ時間合計(秒)": non_j_delay})
    rows.append({"交差点": "全体合計",      "前半台数合計": j_t1 + non_j_t1, "後半台数合計": j_t2 + non_j_t2, "遅れ時間合計(秒)": j_delay + non_j_delay})

    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"✅ 交差点別集計 CSV 出力完了: {output_path}")


def update_comparison_csv():
    """
    実行した制御モードの集計結果を BASE_OUT/制御モード比較.csv に1行追記する。
    交差点別_待ち台数_遅れ時間_集計.csv を読み込み、実行日時・制御モードを付加する。
    """
    import datetime
    summary_path    = os.path.join(OUT_DIR, "交差点別_待ち台数_遅れ時間_集計.csv")
    comparison_path = os.path.join(BASE_OUT, "制御モード比較.csv")

    if not os.path.exists(summary_path):
        print("⚠️ 交差点別集計CSVが見つかりません。比較ファイルの更新をスキップ。")
        return

    df = pd.read_csv(summary_path, encoding="utf-8-sig").fillna("")

    def _get(label):
        row = df[df["交差点"] == label]
        if row.empty:
            return 0, 0, 0
        r = row.iloc[0]
        return (int(r.get("前半台数合計", 0) or 0),
                int(r.get("後半台数合計", 0) or 0),
                int(r.get("遅れ時間合計(秒)", 0) or 0))

    j_t1,  j_t2,  j_delay  = _get("J交差点合計")
    nj_t1, nj_t2, nj_delay = _get("J以外合計")
    al_t1, al_t2, al_delay = _get("全体合計")

    row_data = {
        "実行日時":            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "制御モード":          _MODE_FOLDER_MAP[CONTROL_MODE],
        "J前半台数合計":       j_t1,
        "J後半台数合計":       j_t2,
        "J遅れ時間合計(秒)":   j_delay,
        "J以外前半台数合計":   nj_t1,
        "J以外後半台数合計":   nj_t2,
        "J以外遅れ時間合計(秒)": nj_delay,
        "全体前半台数合計":    al_t1,
        "全体後半台数合計":    al_t2,
        "全体遅れ時間合計(秒)": al_delay,
    }
    for tl in TRAFFIC_LIGHT_CONFIG:
        t1, t2, delay = _get(f"交差点{tl}")
        row_data[f"{tl}_前半台数"] = t1
        row_data[f"{tl}_後半台数"] = t2
        row_data[f"{tl}_遅れ時間(秒)"] = delay

    new_row = pd.DataFrame([row_data])
    if os.path.exists(comparison_path):
        existing = pd.read_csv(comparison_path, encoding="utf-8-sig")
        combined = pd.concat([existing, new_row], ignore_index=True)
    else:
        combined = new_row

    combined.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    print(f"✅ 比較ファイル更新: {comparison_path}")


def plot_phase_duration_log(csv_path):
    import matplotlib
    matplotlib.rcParams['font.family'] = 'Meiryo'
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        print("⚠️ フェーズ時間ログが空または存在しません。")
        return
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if df.empty or "回数" not in df.columns:
        print("⚠️ データがありません or '回数' 列が存在しません。")
        return
    total_rows   = len(df)
    rows_per_hour = total_rows / 24
    tick_labels  = [f"{i:02d}:00" for i in range(25)]
    tick_indices = [int(i * rows_per_hour) for i in range(25)]
    tick_indices = [i for i in tick_indices if i < total_rows]
    tick_labels  = tick_labels[:len(tick_indices)]
    plt.figure(figsize=(12, 5))
    plt.plot(df["回数"], df["パターンA(秒)"], label="パターンA（青）", marker="o")
    plt.xlabel("時刻"); plt.ylabel("フェーズ秒数")
    plt.title("信号J パターンA（青信号）時間の推移（道路1制御）")
    plt.xticks(ticks=tick_indices, labels=tick_labels, rotation=45)
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()


def plot_wasted_green_log(csv_path):
    import matplotlib
    matplotlib.rcParams['font.family'] = 'Meiryo'
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        print("⚠️ 無駄青時間ログが空または存在しません。")
        return
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if df.empty or "回数" not in df.columns:
        print("⚠️ データがありません or '回数' 列が存在しません。")
        return
    total_rows    = len(df)
    rows_per_hour = total_rows / 24
    tick_labels   = [f"{i:02d}:00" for i in range(25)]
    tick_indices  = [int(i * rows_per_hour) for i in range(25)]
    tick_indices  = [i for i in tick_indices if i < total_rows]
    tick_labels   = tick_labels[:len(tick_indices)]
    plt.figure(figsize=(12, 5))
    plt.plot(df["回数"], df["無駄青時間(s)"], marker='o', label="無駄青時間")
    plt.xlabel("時刻（1時間ごと）"); plt.ylabel("無駄青時間（秒）")
    plt.title("信号A 無駄青時間（24:00固定）")
    plt.xticks(ticks=tick_indices, labels=tick_labels, rotation=45)
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()


def plot_combined_phase_and_wasted_log(phase_log_path, wasted_log_path):
    import matplotlib
    from matplotlib.ticker import MaxNLocator
    matplotlib.rcParams['font.family'] = 'Meiryo'
    if not os.path.exists(phase_log_path) or not os.path.exists(wasted_log_path):
        print("⚠️ ログファイルが存在しません。")
        return
    df_phase  = pd.read_csv(phase_log_path,  encoding="utf-8-sig")
    df_wasted = pd.read_csv(wasted_log_path, encoding="utf-8-sig")
    if df_phase.empty or df_wasted.empty:
        print("⚠️ ログファイルの中身が空です。")
        return
    total_rows    = len(df_phase)
    rows_per_hour = total_rows / 24
    tick_labels   = [f"{i:02d}:00" for i in range(25)]
    tick_indices  = [int(i * rows_per_hour) for i in range(25)]
    tick_indices  = [i for i in tick_indices if i < total_rows]
    tick_labels   = tick_labels[:len(tick_indices)]
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.set_xlabel("時刻 (h）"); ax1.set_ylabel("青時間時間（s）", color="tab:blue")
    ax1.plot(df_phase["回数"], df_phase["パターンA(秒)"], color="tab:blue", marker="o", label="フェーズ0")
    ax1.tick_params(axis='y', labelcolor="tab:blue")
    ax1.set_xticks(tick_indices); ax1.set_xticklabels(tick_labels, rotation=45)
    ax1.yaxis.set_major_locator(MaxNLocator(nbins='auto', integer=True))
    ax2 = ax1.twinx()
    ax2.set_ylabel("無駄青時間（s）", color="tab:red")
    ax2.plot(df_wasted["回数"], df_wasted["無駄青時間(s)"], color="tab:red", marker="x", label="無駄青")
    ax2.tick_params(axis='y', labelcolor="tab:red")
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title("青時間と無駄青時間の推移"); fig.tight_layout(); plt.grid(True); plt.show()


def plot_green_duration_log(csv_path):
    import matplotlib
    import datetime
    import numpy as np
    matplotlib.rcParams['font.family'] = 'Meiryo'
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        print("⚠️ 青時間ログが空または存在しません。")
        return
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if "step" not in df.columns or "青時間(秒)" not in df.columns:
        print("⚠️ 必要な列(step, 青時間(秒))が存在しません。")
        return
    plt.figure(figsize=(12, 5))
    df_main  = df[df["種類"] == "主道路"]
    df_minor = df[df["種類"] == "従道路"]
    plt.plot(df_main["step"],  df_main["青時間(秒)"],  marker="o", label="主道路", color="tab:blue")
    plt.plot(df_minor["step"], df_minor["青時間(秒)"], marker="s", label="従道路", color="tab:orange")
    max_step = df["step"].max()
    xticks   = np.arange(0, max_step + 3600, 3600)
    xticklabels = [f"{int(h):02d}:00" for h in range(len(xticks))]
    plt.xticks(xticks, xticklabels, rotation=45)
    plt.xlabel("現実時間 (hh:mm)"); plt.ylabel("青信号時間 [秒]")
    plt.title("主道路・従道路の青信号時間の経時変化")
    plt.grid(True, linestyle="--", alpha=0.7); plt.legend(); plt.tight_layout(); plt.show()


# ============================================================
# 時間窓別集計（全24h と 6〜19時 の両方を出力）
#   per-cycle の queue_delay ログ（vehicle_delay_time方式の待ち台数・遅れ時間）を
#   step→時刻に変換し、集計時間窓で絞った集計と全24h集計を並べて出す。
# ============================================================
def summarize_time_window():
    a_dir = os.path.join(log_dir_path, "交差点A")
    win_label = f"{ANALYSIS_START_HOUR}-{ANALYSIS_END_HOUR}時"
    rows = []

    def _agg(df, col_wait, col_delay, label):
        windows = [
            ("全24h", pd.Series(True, index=df.index)),
            (win_label, df["step"].apply(in_analysis_window)),
        ]
        for win, mask in windows:
            d = df[mask]
            n = len(d)
            ws = float(d[col_wait].sum()) if n else 0.0
            ds = float(d[col_delay].sum()) if n else 0.0
            rows.append({
                "対象": label, "窓": win, "サイクル数": n,
                "待ち台数合計": int(ws), "待ち台数平均": round(ws / n, 3) if n else 0,
                "遅れ時間合計(秒)": int(ds), "遅れ時間平均(秒)": round(ds / n, 3) if n else 0,
            })

    total_path = os.path.join(a_dir, "交差点A_total_queue_delay.csv")
    if os.path.isfile(total_path):
        df = pd.read_csv(total_path, encoding="utf-8-sig")
        _agg(df, "全道路停車車両数合計", "全道路遅れ時間合計", "全道路(J:1/5/6/10)")

    for rid in ["1", "5", "6", "10"]:
        p = os.path.join(a_dir, f"道路{rid}_queue_delay.csv")
        if os.path.isfile(p):
            df = pd.read_csv(p, encoding="utf-8-sig")
            _agg(df, "待ち台数合計", "遅れ時間合計", f"道路{rid}")

    if not rows:
        print("⚠️ 時間窓別集計: 集計対象のログが見つかりませんでした")
        return
    out_df = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, f"時間窓別集計_全24h_と_{ANALYSIS_START_HOUR}-{ANALYSIS_END_HOUR}時.csv")
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"✅ 時間窓別集計を出力: {out_path}")
    for r in rows:
        if r["対象"].startswith("全道路"):
            print(f"   [{r['窓']:>7}] サイクル{r['サイクル数']:>4} / "
                  f"待ち台数合計={r['待ち台数合計']} / 遅れ合計={r['遅れ時間合計(秒)']}s "
                  f"(平均{r['遅れ時間平均(秒)']}s/サイクル)")


# ============================================================
# エントリーポイント
# ============================================================
if __name__ == "__main__":
    run_simulation(sumocfg_path, log_dir_path, control_signal=True)
    postprocess_logs(log_dir_path)
    postprocess_intersection_summary()
    update_comparison_csv()
    summarize_time_window()
    write_signal_delay_summary()
    write_hourly_summary()
    plot_phase_duration_log(phase_log_path)
    plot_wasted_green_log(wasted_green_log_path)
    plot_combined_phase_and_wasted_log(phase_log_path, wasted_green_log_path)
    plot_green_duration_log(green_log_path)
    plot_time_space_diagram()
