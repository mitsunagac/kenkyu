# ============================================================================
# 赤2分割GNN制御の核心部（モデル・予測・delta・実時間計測）
# ----------------------------------------------------------------------------
# 赤3分割_制御コア.py の2分割版。違いは次の3点だけで、他は同じ思想:
#   ① モデルが 2入力2出力（[前半,後半] → [次サイクルの前半,後半]）
#   ② 赤を2等分して数える（RedSplit2Tracker）
#   ③ delta は旧2分割制御（1. 信号制御用_従道路も.py）の規則をそのまま使う
#
# ★サイクル長スケジュール（get_target_cycle / build_scaled_logic）と
#   計測対象エッジ（MEAS_CONFIG）は3分割コアと完全に同じなので、そちらを
#   import して共用する。二重管理して食い違うのを防ぐため。
#
# 使い方（制御スクリプト側）:
#   from 赤2分割_制御コア import (RedSplitPredictionGNN, run_prediction_red2,
#                                  delta_red2, load_road_model, RedSplit2Tracker)
#   model1, scaler1 = load_road_model("道路1_分散評価あり")
#   pred = run_prediction_red2(wait_9x2, target_node_index=0,
#                              model=model1, scaler_wait=scaler1, edge_index=ei)
#   d = delta_red2(pred)
# ============================================================================
import os
import math
import importlib.util
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import joblib

CYCLE_LEN = 2      # 赤2分割（前半 / 後半）
HORIZON   = 2      # 出力 t3,t4
ALPHA     = 2      # 対象道路ノードの強調（学習時と一致）

# --- 3分割コアから共用部分を読み込む（サイクル長・計測エッジ定義） ---
_core3_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "赤3分割_制御コア.py")
_spec3 = importlib.util.spec_from_file_location("red3_core_shared", _core3_path)
_red3 = importlib.util.module_from_spec(_spec3)
_spec3.loader.exec_module(_red3)

MEAS_CONFIG           = _red3.MEAS_CONFIG            # 9道路の tl / green / yellow / edges / divisor
MODEL_NODE_ORDER      = _red3.MODEL_NODE_ORDER       # node i = road i+1
STOP_SPEED_THRESHOLD  = _red3.STOP_SPEED_THRESHOLD
BASE_CYCLE_LENGTH     = _red3.BASE_CYCLE_LENGTH      # 120（ネットの基準サイクル）
PREP_CYCLE_LENGTH     = _red3.PREP_CYCLE_LENGTH
CYCLE_SCHEDULE_BY_HOUR = _red3.CYCLE_SCHEDULE_BY_HOUR
get_target_cycle      = _red3.get_target_cycle
build_scaled_logic    = _red3.build_scaled_logic

RED_SPLIT_COUNT = 2

# 研究/制御/新環境用/ から 研究/ まで3つ上がり、研究/予測/予測モデル/… を指す
_PRED_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "予測", "予測モデル", "1. 赤時間2分割",
)


# ---- 学習時と同一アーキテクチャ（★予測モデル作成_待ち台数_赤2分割.py と一致必須）----
class RedSplitPredictionGNN(torch.nn.Module):
    def __init__(self, num_node_features=CYCLE_LEN, horizon=HORIZON, dropout_rate=0.2):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, 128)
        self.skip  = nn.Linear(num_node_features, 128)
        self.lin_out = nn.Linear(128, horizon)
        self.dropout_rate = dropout_rate

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        h = F.relu(self.conv3(h, edge_index))
        h = h + self.skip(x)
        return self.lin_out(h)


def load_road_model(model_dir_name, device=None):
    """予測モデル/1. 赤時間2分割/<model_dir_name>/ からモデルとスケーラーを読み込む。
    例: load_road_model("道路1_分散評価あり") / load_road_model("道路1_分散t4のみ")"""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = os.path.join(_PRED_ROOT, model_dir_name)
    model = RedSplitPredictionGNN().to(device)
    model.load_state_dict(torch.load(os.path.join(d, "GCN_epoch_best_model.pth"),
                                     map_location=device))
    model.eval()
    scaler = joblib.load(os.path.join(d, "scaler_wait.pkl"))
    return model, scaler


def run_prediction_red2(wait_9x2, target_node_index, model, scaler_wait, edge_index):
    """全9道路の当該サイクル2分割 [9,2]（実台数）→ 対象道路の次サイクル [t3,t4]（実台数）。
    学習と同じく: 9道路方向に MinMax、対象ノードのみ ×ALPHA、対象列のみ逆正規化。"""
    device = next(model.parameters()).device
    arr = np.asarray(wait_9x2, dtype=float)                    # [9,2]
    scaled = np.stack([scaler_wait.transform(arr[:, k].reshape(1, -1)).reshape(-1)
                       for k in range(CYCLE_LEN)], axis=1)      # [9,2]
    x = torch.tensor(scaled, dtype=torch.float).to(device)
    x[target_node_index] *= ALPHA
    with torch.no_grad():
        out = model(x, edge_index.to(device)).cpu().numpy()    # [9,2] 正規化
    o = out[target_node_index]                                  # [2]
    mn = scaler_wait.data_min_[target_node_index]
    mx = scaler_wait.data_max_[target_node_index]
    return o * (mx - mn) + mn                                   # [t3,t4] 実台数


# ================== delta（需要重み付け方式・現行） ==================
# ギャップ感応と同じく「初期状態 = BASE − 10s」から始め、予測台数に応じて足していく。
#
#     delta = -10 + (前半予測台数 × x1 + 後半予測台数 × x2) × 2
#
#   ・重み付き台数 0台  → delta = -10（= BASE − 10s、ギャップ感応の最小青と同じ位置）
#   ・重み付き台数 5台  → delta =   0（= BASE ちょうど）
#   ・重み付き台数 10台 → delta = +10（= BASE + 10s、上限。これ以上はクランプ）
#   ・x1 > x2（前半を重く見る）、0 < x1, x2 < 1
#   ・予測値は負になることがあるので 0 で下限を切ってから使う
#
# 参考（2026-08-03 の24時間走行の予測分布）:
#   道路1: 前半 平均3.51台 / 後半 平均2.37台
#   道路5: 前半 平均1.64台 / 後半 平均0.45台
#   x1=0.9, x2=0.6 のとき delta 平均は 道路1 で -1.2s、道路5 で -6.5s
GREEN_DEV_START = -10.0   # 予測台数0のときの BASE からのずれ（= ギャップ感応の最小青と同じ）
X1_DEFAULT = 0.9          # 前半（t3）の重み
X2_DEFAULT = 0.6          # 後半（t4）の重み


X1_MAX_PEAK = 1.5   # 交通量の多い時間帯に限り x1 はここまで上げてよい


def validate_weights(x1, x2, peak=False):
    """x1 > x2 を必須とし、x1 の上限は通常 1.0 未満、ピーク時間帯のみ X1_MAX_PEAK まで許す。
    x2 は常に 0 < x2 < 1。"""
    hi = X1_MAX_PEAK if peak else 1.0
    label = f"（ピーク時間帯は {X1_MAX_PEAK} まで可）" if peak else ""
    if not (0.0 < x1 <= hi if peak else 0.0 < x1 < 1.0):
        raise ValueError(f"x1 は 0 < x1 {'≤' if peak else '<'} {hi} の範囲で指定してください"
                         f"（x1={x1}）{label}")
    if not (0.0 < x2 < 1.0):
        raise ValueError(f"x2 は 0 < x2 < 1 の範囲で指定してください（x2={x2}）")
    if not (x1 > x2):
        raise ValueError(f"x1 > x2 としてください（x1={x1}, x2={x2}）")
    return True


def delta_red2_weighted(t34, x1=X1_DEFAULT, x2=X2_DEFAULT, clamp=None):
    """t34=[t3,t4]（次サイクルの予測台数）→ delta(int, ±clamp)。
    delta = GREEN_DEV_START + (t3*x1 + t4*x2) * 2 を丸めて ±clamp に収める。"""
    t = np.asarray(t34, dtype=float)
    t3 = max(0.0, float(t[0]))
    t4 = max(0.0, float(t[1]))
    cl = DELTA_CLAMP if clamp is None else clamp
    delta = GREEN_DEV_START + (t3 * x1 + t4 * x2) * 2.0
    return int(round(max(-cl, min(delta, cl))))


# ================== delta（旧2分割制御の規則・比較用に残す） ==================
# 1. 信号制御用_従道路も.py の adjust_signal_A_based_on_prediction より:
#     t3 > t4 → delta =  t3 * 2   （前半が多い = 早く捌きたい → 青を延長）
#     t3 < t4 → delta = -t4 * 2   （後半が多い = まだ来る    → 青を短縮）
#     t3 = t4 → delta =  0
#     最後に ±DELTA_CLAMP で頭打ち。
# ※ 3分割の案1と同じ発想だが、区間が2つなので「中盤最大→0」の分岐は無い。
#   デッドバンドも旧仕様どおり既定 0（= 常に動かす）。3分割の案1に揃えたい場合は
#   環境変数 R2_DEADBAND=2 のように制御スクリプト側から DEADBAND を差し替える。
DELTA_CLAMP = 10
DEADBAND    = 0    # 前半と後半の差がこの台数未満なら delta=0（0 で無効）


def delta_red2(t34, deadband=None, clamp=None):
    """t34=[t3,t4]（実台数）→ delta(int, ±clamp)。"""
    t = np.asarray(t34, dtype=float)
    t3, t4 = float(t[0]), float(t[1])
    db = DEADBAND if deadband is None else deadband
    cl = DELTA_CLAMP if clamp is None else clamp

    if abs(t3 - t4) < db:
        return 0
    if t3 > t4:
        delta = t3 * 2
    elif t3 < t4:
        delta = -t4 * 2
    else:
        delta = 0
    return int(max(-cl, min(delta, cl)))


# ============================================================================
# 赤2分割の実時間計測
# ----------------------------------------------------------------------------
# 3分割の RedSplit3Tracker と同一方式（初停車ビン）で、分割数だけ 2 にしたもの。
#   ・赤サイクル = yellow_phase 終了 〜 次の green_phase 開始（フェーズ番号ベース）
#   ・赤中: 対象エッジで初めて speed≤0 になった車の「赤開始からの経過時刻」を記録
#   ・赤終了: 赤時間を2等分し、各区間で初停車した台数を数える（÷divisor, ceil）
#
# ★旧2分割制御（1. 信号制御用_従道路も.py）は「赤開始から固定オフセット
#   （道路1: +28s/+56s、道路5: +45s/+90s）の瞬間に停車中の台数を数える」方式だった。
#   固定オフセットはサイクル長130s固定を前提とした値で、時間帯別サイクル
#   （100/110/130s）では赤の別の位置を測ってしまうため、実測赤を2等分する方式に変更した。
#   赤の間は車が列から出ていかないので「その時点で停車中の台数」≒「そこまでに初停車した
#   台数」であり、定義の差は前サイクルの残留車と微速前進車の扱いだけ。
# ============================================================================
class RedSplit2Tracker:
    """全9道路の赤2分割を実時間で計測する。毎ステップ step() を呼ぶ。
    road の赤が終了したら latest[road] に [前半, 後半] が入り、finalized に road_id を返す。"""

    def __init__(self, config=MEAS_CONFIG, split_count=RED_SPLIT_COUNT,
                 stop_speed=STOP_SPEED_THRESHOLD):
        self.cfg = config
        self.split = split_count
        self.stop_speed = stop_speed
        self.tls = sorted({c["tl"] for c in config.values()})
        self.active = {}        # road_id -> {"red_start", "recorded":set, "stop_times":[]}
        self.latest = {}        # road_id -> [前半, 後半]（最新確定サイクル）
        self.latest_N = {}      # road_id -> 最大間隔発生台数目N（参考値・現状未使用）
        self.prev_phase = {}    # tl_id -> 前ステップのフェーズ番号

    def _finalize(self, road_id, red_end):
        a = self.active.pop(road_id, None)
        if a is None:
            return None
        red_dur = red_end - a["red_start"]
        counts = [0] * self.split
        if red_dur > 0:
            half = red_dur / self.split
            for st in a["stop_times"]:
                idx = int(st // half)
                if idx >= self.split:
                    idx = self.split - 1           # 境界(=赤終了ちょうど)は最終区間へ
                counts[idx] += 1
        div = self.cfg[road_id]["divisor"]
        counts = [math.ceil(c / div) for c in counts]
        self.latest[road_id] = counts

        # 最大間隔発生台数目 N（到着間隔系の制御を後から足すとき用に計算だけしておく）
        st = a["stop_times"]
        if st:
            intervals = [st[0]] + [st[i] - st[i - 1] for i in range(1, len(st))]
            max_index = max(range(len(intervals)), key=lambda k: intervals[k])
            self.latest_N[road_id] = int(round((max_index + 1) / 3))
        else:
            self.latest_N[road_id] = 0
        return counts

    def step(self, traci, sim_time, prep_time=0):
        """SUMOの1ステップ後に呼ぶ。戻り: このステップで赤終了し確定した road_id のリスト。"""
        cur = {tl: traci.trafficlight.getPhase(tl) for tl in self.tls}
        finalized = []

        for road_id, c in self.cfg.items():
            tl = c["tl"]
            cur_p = cur[tl]
            prev_p = self.prev_phase.get(tl)

            # 赤開始: その道路の黄が終わった瞬間
            if (sim_time >= prep_time and prev_p == c["yellow"] and cur_p != c["yellow"]):
                self.active[road_id] = {"red_start": sim_time,
                                        "recorded": set(), "stop_times": []}

            # 赤終了: その道路が青になった瞬間 → 2分割確定
            if (prev_p is not None and prev_p != c["green"] and cur_p == c["green"]):
                if road_id in self.active:
                    self._finalize(road_id, sim_time)
                    finalized.append(road_id)

            # 赤中: 対象エッジで初めて停車した車を記録
            a = self.active.get(road_id)
            if a is not None:
                st_after = sim_time - a["red_start"]
                for edge in c["edges"]:
                    for veh in traci.edge.getLastStepVehicleIDs(edge):
                        if veh in a["recorded"]:
                            continue
                        if traci.vehicle.getSpeed(veh) <= self.stop_speed:
                            a["recorded"].add(veh)
                            a["stop_times"].append(st_after)

        for tl in self.tls:
            self.prev_phase[tl] = cur[tl]
        return finalized

    def model_input(self):
        """全9道路の最新2分割から [9,2] を組む。1つでも未確定なら None。"""
        if not all(r in self.latest for r in MODEL_NODE_ORDER):
            return None
        return np.array([self.latest[r] for r in MODEL_NODE_ORDER], dtype=float)  # [9,2]
