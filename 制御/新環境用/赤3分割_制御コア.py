# ============================================================================
# 赤3分割GNN制御の核心部（モデル・予測・delta全案）
# ----------------------------------------------------------------------------
# ★実モデル（予測モデル/3. 赤時間3分割/道路{1,5}_分散評価あり）と実データで検証済み。
#   ・モデル読み込み（load_state_dict）成功 = 学習アーキと一致
#   ・予測 [9,3] → [t4,t5,t6] 動作
#   ・delta 全案（1/2A/2B/3/4）が設計書の実例と一致、案4==案2-B(道路1)も確認
#
# 使い方（制御スクリプト側）:
#   from 赤3分割_制御コア import (RedSplitPredictionGNN, run_prediction_red3,
#                                  delta_variant, load_road_model)
#   model1, scaler1 = load_road_model("道路1_分散評価あり")
#   pred = run_prediction_red3(wait_9x3, target_node_index=0,
#                              model=model1, scaler_wait=scaler1, edge_index=ei)
#   d = delta_variant(pred, road_id="1", variant="2A")
# ============================================================================
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import joblib

CYCLE_LEN = 3      # 赤3分割
HORIZON   = 3      # 出力 t4,t5,t6
ALPHA     = 2      # 対象道路ノードの強調（学習時と一致）

# 研究/制御/新環境用/ から 研究/ まで3つ上がり、研究/予測/予測モデル/… を指す。
# （フォルダを 計測/予測/制御 の3構成に整理した際に予測モデルが 研究/予測/ 配下へ移動したため、
#   旧パス 研究/予測モデル/ のままだと FileNotFoundError になっていた。2026-08-02 修正）
_PRED_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "予測", "予測モデル", "3. 赤時間3分割",
)


# ---- 学習時と同一アーキテクチャ（★予測モデル作成_待ち台数_赤3分割.py と一致必須）----
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
    """予測モデル/3. 赤時間3分割/<model_dir_name>/ からモデルとスケーラーを読み込む。"""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = os.path.join(_PRED_ROOT, model_dir_name)
    model = RedSplitPredictionGNN().to(device)
    model.load_state_dict(torch.load(os.path.join(d, "GCN_epoch_best_model.pth"),
                                     map_location=device))
    model.eval()
    scaler = joblib.load(os.path.join(d, "scaler_wait.pkl"))
    return model, scaler


def run_prediction_red3(wait_9x3, target_node_index, model, scaler_wait, edge_index):
    """全9道路の当該サイクル3分割 [9,3]（実台数）→ 対象道路の次サイクル [t4,t5,t6]（実台数）。
    学習と同じく: 9道路方向に MinMax、対象ノードのみ ×ALPHA、対象列のみ逆正規化。"""
    device = next(model.parameters()).device
    arr = np.asarray(wait_9x3, dtype=float)                    # [9,3]
    scaled = np.stack([scaler_wait.transform(arr[:, k].reshape(1, -1)).reshape(-1)
                       for k in range(CYCLE_LEN)], axis=1)      # [9,3]
    x = torch.tensor(scaled, dtype=torch.float).to(device)
    x[target_node_index] *= ALPHA
    with torch.no_grad():
        out = model(x, edge_index.to(device)).cpu().numpy()    # [9,3] 正規化
    o = out[target_node_index]                                  # [3]
    mn = scaler_wait.data_min_[target_node_index]
    mx = scaler_wait.data_max_[target_node_index]
    return o * (mx - mn) + mn                                   # [t4,t5,t6] 実台数


# ================== delta 全案（設計書 2026-07-23 確定） ==================
WEIGHTS = {"A": np.array([1.5, 1.0, 0.5]),   # 前半重視
           "B": np.array([0.5, 1.0, 1.5])}   # 後半重視
# W_ref（重み付き需要の中央値 / 合計中央値）: [道路][案]
#  ・2A/2B … 重み付き需要の中央値（各重みで取り直したもの）
#  ・3     … 重みA × 合計中央値（重みなし t4+t5+t6 の中央値）
#  ※ 案4（重みB × 合計中央値）は案2-Bと数値完全一致のため廃止（2026-07-23）。
W_REF = {
    "1": {"2A": 4.0, "2B": 5.0, "3": 5.0},
    "5": {"2A": 2.5, "2B": 2.0, "3": 2.0},
}
DEADBAND = 2   # 案1のデッドバンド(台): 最大区間が2番目より2台以上多いときだけ動かす
DELTA_CLAMP = 10


def delta_variant(t456, road_id, variant):
    """t456=[t4,t5,t6]（実台数）→ delta(int, ±DELTA_CLAMP)。
    variant: '1'(最大区間), '2A','2B'(重み付き需要−重み付き中央値), '3'(重みA−合計中央値)。"""
    t = np.asarray(t456, dtype=float)
    road_id = str(road_id)
    if variant not in ("1", "2A", "2B", "3"):
        raise ValueError(f"未知の制御案: {variant}（有効: 1/2A/2B/3）")

    if variant == "1":
        order = np.argsort(t)[::-1]
        if t[order[0]] - t[order[1]] < DEADBAND:
            return 0
        idx = int(order[0])                     # 0=t4前半 / 1=t5中盤 / 2=t6後半
        if idx == 0:
            return int(max(-DELTA_CLAMP, min(t[0] * 2, DELTA_CLAMP)))    # 前半最大→延長
        elif idx == 2:
            return int(max(-DELTA_CLAMP, min(-t[2] * 2, DELTA_CLAMP)))   # 後半最大→短縮
        return 0                                                         # 中盤最大→0

    w = WEIGHTS["A"] if variant in ("2A", "3") else WEIGHTS["B"]
    W = float(t @ w)
    delta = (W - W_REF[road_id][variant]) * 2
    return int(max(-DELTA_CLAMP, min(delta, DELTA_CLAMP)))


# ============================================================================
# 赤3分割の実時間計測（学習データ生成プログラムと同一方式）
# ----------------------------------------------------------------------------
# ★★赤時間3分割計測_北南20%削減全日_2026-07-17.py の record_stops + write_red_split を移植。
#   ・赤サイクル = yellow_phase 終了 〜 次の green_phase 開始（フェーズ番号ベース・可変長対応）
#   ・赤中: 対象エッジで初めて speed≤0 になった車の「赤開始からの経過時刻」を記録
#   ・赤終了: 赤時間を3等分し、各区間で初停車した台数を数える（÷divisor, ceil）
#   ・待ち行列長(累積)ではなく「初停車ビン(非累積)」を数える点が既存制御と異なる。
# ============================================================================

# 9道路の計測設定（計測プログラムの TRAFFIC_LIGHT_CONFIG + ROAD_PHASE_TARGETS より）。
# road_id はモデルのノード順（道路1=node0 … 道路9=node8）に対応。
MEAS_CONFIG = {
    "1": {"tl": "J", "green": 0, "yellow": 1, "edges": ["E24", "-E23", "-E33"], "divisor": 2},
    "2": {"tl": "E", "green": 0, "yellow": 1, "edges": ["E40", "-E22"],          "divisor": 2},
    "3": {"tl": "F", "green": 0, "yellow": 1, "edges": ["E3", "E1", "-E34", "E35"], "divisor": 2},
    "4": {"tl": "I", "green": 0, "yellow": 1, "edges": ["E57"],                   "divisor": 2},
    "5": {"tl": "J", "green": 5, "yellow": 6, "edges": ["E37", "E36"],            "divisor": 2},
    "6": {"tl": "J", "green": 5, "yellow": 6, "edges": ["-E25", "-E69"],          "divisor": 2},
    "7": {"tl": "K", "green": 0, "yellow": 1, "edges": ["155398822#14", "155398822#13"], "divisor": 2},
    "8": {"tl": "M", "green": 0, "yellow": 1, "edges": ["E31", "E30"],            "divisor": 2},
    "9": {"tl": "N", "green": 0, "yellow": 1, "edges": ["E20", "E19"],            "divisor": 2},
}
RED_SPLIT_COUNT = 3
STOP_SPEED_THRESHOLD = 0.0
MODEL_NODE_ORDER = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]   # node i = road i+1


class RedSplit3Tracker:
    """全9道路の赤3分割を実時間で計測する。毎ステップ step() を呼ぶ。
    road の赤が終了したら latest[road] に [c1,c2,c3] が入り、finalized に road_id を返す。"""

    def __init__(self, config=MEAS_CONFIG, split_count=RED_SPLIT_COUNT,
                 stop_speed=STOP_SPEED_THRESHOLD):
        self.cfg = config
        self.split = split_count
        self.stop_speed = stop_speed
        self.tls = sorted({c["tl"] for c in config.values()})
        self.active = {}        # road_id -> {"red_start", "recorded":set, "stop_times":[]}
        self.latest = {}        # road_id -> [c1,c2,c3]（最新確定サイクル）
        self.latest_N = {}      # road_id -> 最大間隔発生台数目N（最新確定サイクル・予測ギャップ制御用）
        self.prev_phase = {}    # tl_id -> 前ステップのフェーズ番号

    def _finalize(self, road_id, red_end):
        a = self.active.pop(road_id, None)
        if a is None:
            return None
        red_dur = red_end - a["red_start"]
        counts = [0] * self.split
        if red_dur > 0:
            third = red_dur / self.split
            for st in a["stop_times"]:
                idx = int(st // third)
                if idx >= self.split:
                    idx = self.split - 1           # 境界(=赤終了ちょうど)は最終区間へ
                counts[idx] += 1
        div = self.cfg[road_id]["divisor"]
        counts = [math.ceil(c / div) for c in counts]
        self.latest[road_id] = counts

        # 最大間隔発生台数目 N（予測ギャップ制御用）: stop_times の車間が最大の位置(÷3四捨五入)
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

            # 赤終了: その道路が青になった瞬間 → 3分割確定
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
        """全9道路の最新3分割から [9,3] を組む。1つでも未確定なら None。"""
        if not all(r in self.latest for r in MODEL_NODE_ORDER):
            return None
        return np.array([self.latest[r] for r in MODEL_NODE_ORDER], dtype=float)  # [9,3]


# ============================================================================
# 時間帯別サイクル長制御（学習データ生成プログラムと同一）
# ----------------------------------------------------------------------------
# ★★赤時間3分割計測_…_2026-07-17.py の build_scaled_logic / get_target_cycle を移植。
#   120s基準ネットの青フェーズ(≥10s)に ΔC を均等配分してサイクル長を作る。
#   黄・全赤・右折矢印(短フェーズ)は固定。J交差点の制御delta はこれに加算される。
# ============================================================================
BASE_CYCLE_LENGTH = 120
PREP_CYCLE_LENGTH = 100
GREEN_PHASE_MIN_DURATION = 10
CYCLE_SCHEDULE_BY_HOUR = [
    (0, 7, 100),    # 0〜7時
    (7, 8, 130),    # 7〜8時(朝ピーク)
    (8, 17, 110),   # 8〜17時
    (17, 19, 130),  # 17〜19時(夕ピーク)
    (19, 24, 100),  # 19〜24時
]


def get_target_cycle(sim_time, warmup_time):
    """シミュレーション時刻(s)から目標サイクル長(s)を返す。"""
    if sim_time < warmup_time:
        return PREP_CYCLE_LENGTH
    hour = (sim_time - warmup_time) / 3600.0
    for start_hour, end_hour, cycle in CYCLE_SCHEDULE_BY_HOUR:
        if start_hour <= hour < end_hour:
            return cycle
    return CYCLE_SCHEDULE_BY_HOUR[-1][2]


def build_scaled_logic(traci, base_logic, target_cycle):
    """基準プログラムの青フェーズ(≥GREEN_PHASE_MIN_DURATION)に (target-base)/n を均等配分し、
    目標サイクル長のLogicを生成する。フェーズ数・状態・順序・番号は不変。"""
    phases = base_logic.phases
    base_cycle = sum(ph.duration for ph in phases)
    green_idxs = [i for i, ph in enumerate(phases) if ph.duration >= GREEN_PHASE_MIN_DURATION]
    if not green_idxs:
        return None
    per_green_delta = (target_cycle - base_cycle) / len(green_idxs)
    new_phases = []
    for i, ph in enumerate(phases):
        next_phases = getattr(ph, "next", ())
        name = getattr(ph, "name", "")
        if i in green_idxs:
            nd = ph.duration + per_green_delta
            new_phases.append(traci.trafficlight.Phase(nd, ph.state, nd, nd, next_phases, name))
        else:
            new_phases.append(traci.trafficlight.Phase(
                ph.duration, ph.state, ph.minDur, ph.maxDur, next_phases, name))
    return traci.trafficlight.Logic(
        base_logic.programID, base_logic.type,
        base_logic.currentPhaseIndex, new_phases, base_logic.subParameter)
