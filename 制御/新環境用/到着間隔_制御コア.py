# ============================================================================
# 予測ギャップ感応制御の核心部（到着間隔同時予測モデル W・N）
# ----------------------------------------------------------------------------
# ○ 信号制御_mix_到着間隔.py の DualPredictionGNN / run_prediction_dual /
#   compute_max_interval_order を移植し、事前ロードして毎サイクル呼べる形にする。
#
#   ・DualPredictionGNN: 入力 [9,2]=[待ち台数, 間隔台数目] → 出力 [W(t+1), N(t+1)]
#   ・N(最大間隔発生台数目) = 赤中で一番大きい車間が空く「何台目」(÷3・四捨五入)
#   ・間隔Nの現在値は RedSplit3Tracker の stop_times（初停車時刻列）から直接算出できる
#     （別途の到着間隔計測は不要）。
# ============================================================================
import os
import numpy as np
import torch
import torch.nn as nn
import joblib

DUAL_ALPHA = 2
_PRED_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "予測モデル", "2. 到着間隔_同時予測",
)


class DualPredictionGNN(torch.nn.Module):
    def __init__(self, num_node_features=2):
        super().__init__()
        from torch_geometric.nn import GCNConv
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, 128)
        self.skip  = nn.Linear(num_node_features, 128)
        self.lin_wait = nn.Linear(128, 1)
        self.lin_int  = nn.Linear(128, 1)

    def forward(self, x, edge_index):
        h = torch.relu(self.conv1(x, edge_index))
        h = torch.relu(self.conv2(h, edge_index))
        h = torch.relu(self.conv3(h, edge_index))
        h = h + self.skip(x)
        wait = self.lin_wait(h)
        intv = self.lin_int(h)
        return torch.cat([wait, intv], dim=1)   # [num_nodes, 2]


# 到着間隔モデルの道路→ノード対応（道路1=node0, 道路5=node4）
DUAL_ROADS = {
    "1": {"dir": "道路1", "node": 0},
    "5": {"dir": "道路5", "node": 4},
}


def load_dual_model(road_id, device=None):
    """予測モデル/2. 到着間隔_同時予測/道路{road}/ からモデルとスケーラー2種を読み込む。"""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = os.path.join(_PRED_ROOT, DUAL_ROADS[road_id]["dir"])
    model = DualPredictionGNN(num_node_features=2).to(device)
    model.load_state_dict(torch.load(os.path.join(d, "GCN_epoch_best_model.pth"),
                                     map_location=device))
    model.eval()
    scaler_wait = joblib.load(os.path.join(d, "scaler_wait.pkl"))
    scaler_int  = joblib.load(os.path.join(d, "scaler_interval.pkl"))
    return model, scaler_wait, scaler_int


def predict_WN(wait_9, interval_value, model, scaler_wait, scaler_int,
               edge_index, target_node_index, alpha=DUAL_ALPHA):
    """全9道路の待ち台数 + 対象道路の現在間隔Nから、次サイクルの (W_pred, N_pred) を返す。
    ○ 信号制御_mix_到着間隔.py の run_prediction_dual と同一処理（事前ロード版）。"""
    device = next(model.parameters()).device
    w = np.array(wait_9, dtype=float).reshape(1, -1)
    w_scaled = scaler_wait.transform(w).reshape(-1)                       # [9]
    i_scaled = float(scaler_int.transform([[float(interval_value)]]).reshape(-1)[0])

    col_wait = torch.tensor(w_scaled, dtype=torch.float).unsqueeze(1)     # [9,1]
    col_int  = torch.zeros((col_wait.shape[0], 1), dtype=torch.float)     # [9,1]
    col_int[target_node_index, 0] = i_scaled
    col_wait[target_node_index] *= alpha
    x = torch.cat([col_wait, col_int], dim=1).to(device)                 # [9,2]

    with torch.no_grad():
        out = model(x, edge_index.to(device)).cpu().numpy()              # [9,2]
    w_min0 = scaler_wait.data_min_[target_node_index]
    w_max0 = scaler_wait.data_max_[target_node_index]
    i_min, i_max = scaler_int.data_min_[0], scaler_int.data_max_[0]
    W_pred = out[target_node_index, 0] * (w_max0 - w_min0) + w_min0
    N_pred = out[target_node_index, 1] * (i_max  - i_min) + i_min
    return float(W_pred), float(N_pred)


def max_interval_order(stop_times):
    """初停車時刻の列 stop_times（赤開始からの経過秒・記録順）から
    最大間隔発生台数目 N を返す（compute_max_interval_order と同一: ÷3・四捨五入）。
    ・interval[0] = stop_times[0]、interval[i>0] = stop_times[i]-stop_times[i-1]
    ・N = round((argmax(interval)+1) / 3)"""
    if not stop_times:
        return 0
    st = list(stop_times)
    intervals = [st[0]] + [st[i] - st[i - 1] for i in range(1, len(st))]
    max_index = max(range(len(intervals)), key=lambda k: intervals[k])
    return int(round((max_index + 1) / 3))
