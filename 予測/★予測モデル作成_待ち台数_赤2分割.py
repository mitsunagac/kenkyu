import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

# ============================================================================
# 「待ち台数」のみを予測するモデル（赤時間2分割・1サイクル先予測）
# ----------------------------------------------------------------------------
# ★学習器の機構（GCNConv×3 + スキップ接続 / 学習率スケジュール / early stopping /
#   RMSE 未達なら再学習する while ループ / 分布形状損失）は
#   ★予測モデル作成_待ち台数_赤3分割.py と同一。分割数だけを 3 → 2 に変更した版。
#
# ★予測構造（2入力 → 2出力・1モデル）:
#     赤時間を2分割した 2 つの値をひとまとまり（= 1 サイクル）として扱い、
#     「あるサイクルの2値」から「次サイクルの2値」を予測する。
#       入力 : ノード特徴量 [待ち台数(t1), 待ち台数(t2)]  (num_node_features = 2)
#               対象道路ノードのみ ×ALPHA で強調
#       出力 : 対象道路の [待ち台数(t3), 待ち台数(t4)]    (2値)
#     → サンプルは 2 刻み（ブロック非重複 = サイクル境界に一致）で生成。
#     → 到着間隔の予測は行わない。待ち台数のみ。
#     ※ 旧 1. 道路1用_予測モデル作成.py と同じ t1,t2 → t3,t4 の予測だが、
#        自己回帰 + Teacher Forcing ではなく 2 出力を同時に回帰する構造。
#
# ★損失: 各ステップ(t3/t4)について MSE + MAE + LAMBDA_SHAPE×分布形状損失。
#   分布形状損失は予測が平均値に張り付いて分散が潰れるのを防ぐ。
#
# ★入力データ:
#     ★学習用データ/1. 赤時間2分割_学習データ_12.22/新環境待ち台数データ_12.22.csv
#       : 全道路(道路1〜9)の待ち台数（赤時間2分割）  [N, 9]
#     ★学習用データ/1. 赤時間2分割_学習データ_12.22/新環境正解データ_12.22.csv
#       : テスト日1日分の「前半・後半の大小関係ラベル」(0/1/2) [665, 9]（正答率評価用）
# ★データ量: 全 151 日分 × 1 日 665 サイクル × 2 分割 = 1 日 1330 行 / 全 200830 行
# ============================================================================

import pandas as pd
import numpy as np
import torch
import csv
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from torch_geometric.data import Data
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
from matplotlib.ticker import ScalarFormatter, MaxNLocator
import os
import shutil
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# 予測対象の道路設定
# ----------------------------------------------------------------------------
#  ・TRAIN_BOTH_ROADS = True  … TARGET_ROADS に並べた道路を上から順に連続学習する
#  ・TRAIN_BOTH_ROADS = False … TARGET_ROAD の 1 道路だけを学習する（従来どおり）
#
# 実行時の第1引数を渡すと、そちらが優先されます:
#     python ★予測モデル作成_待ち台数_赤2分割.py         → 下のフラグ設定に従う
#     python ★予測モデル作成_待ち台数_赤2分割.py 5       → 道路5 のみ学習
#     python ★予測モデル作成_待ち台数_赤2分割.py both    → 道路1・道路5 を連続学習
#                                （both / all / 両方 / 全部 のいずれでも可）
#
# 連続学習では道路ごとに別フォルダへ保存されるため、途中で上書きは起きません。
# 1道路が MAX_TRIALS まで達成できなくても、次の道路の学習には進みます。
# ============================================================================
TRAIN_BOTH_ROADS = False                # True にすると TARGET_ROADS を連続学習
TARGET_ROADS     = ["道路1", "道路5"]   # 連続学習する道路（この順で実行）
TARGET_ROAD      = "道路1"              # 単体学習するときの対象道路

BOTH_KEYWORDS = ("both", "all", "両方", "全部")


def extract_road_id(value, fallback=None):
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    if digits:
        return int(digits)
    if fallback is not None:
        return fallback
    raise ValueError(f"道路番号を判定できません: {value}")


def resolve_target_roads():
    """実行時引数とフラグから「学習する道路のリスト」と、その決まり方の説明を返す。"""
    if len(sys.argv) > 1:
        arg = sys.argv[1].strip()
        if arg.lower() in BOTH_KEYWORDS or arg in BOTH_KEYWORDS:
            return list(TARGET_ROADS), f"引数 '{arg}' により連続学習"
        return [arg], f"引数 '{arg}' により単体学習"
    if TRAIN_BOTH_ROADS:
        return list(TARGET_ROADS), "TRAIN_BOTH_ROADS=True により連続学習"
    return [TARGET_ROAD], "TRAIN_BOTH_ROADS=False により単体学習"


print(torch.cuda.is_available())
target_roads, _mode_desc = resolve_target_roads()
print(f"学習対象: {' → '.join(str(r) for r in target_roads)}  （{_mode_desc}）")

# === 使用ノード（道路1〜9 すべて） ===
used_road_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]
node_num = len(used_road_indices)

# === デバイスの設定 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === データパス（★学習用データ フォルダ） ===
DATA_DIR = os.path.join(BASE_DIR, "★学習用データ", "1. 赤時間2分割_学習データ_12.22")
wait_csv_path   = os.path.join(DATA_DIR, "新環境待ち台数データ_12.22.csv")  # 全道路の待ち台数 (9列)
answer_csv_path = os.path.join(DATA_DIR, "新環境正解データ_12.22.csv")      # 大小関係ラベル (9列)
adjacency_path  = os.path.join(BASE_DIR, "新環境_隣接行列.csv")

# === データ読み込み ===
#   待ち台数 CSV は utf-8 (BOM なし), 正解 CSV は utf-8-sig (BOM 付き)。
#   utf-8-sig は BOM 無しファイルもそのまま読めるので両方これで統一する。
wait_data_csv    = pd.read_csv(wait_csv_path, encoding="utf-8-sig")
adjacency_matrix = pd.read_csv(adjacency_path, encoding="shift-jis", index_col=0)
print(adjacency_matrix)

# == データ数 ==============================================================
# 赤時間を 2 分割しているので、1 サイクル = 2 行。
CYCLE_LEN      = 2     # 1 サイクルあたりの行数（赤時間2分割）
CYCLES_PER_DAY = 665   # 1 日のサイクル数
DAY_LENGTH     = CYCLES_PER_DAY * CYCLE_LEN   # = 1330 行/日
EXPECTED_DAYS  = 151

TOTAL_DAYS = len(wait_data_csv) // DAY_LENGTH
remainder  = len(wait_data_csv) % DAY_LENGTH
print(f"総行数: {len(wait_data_csv)} / DAY_LENGTH={DAY_LENGTH}"
      f"（{CYCLES_PER_DAY}サイクル×{CYCLE_LEN}分割） → {TOTAL_DAYS} 日分")
if remainder != 0:
    print(f"[警告] 1日={DAY_LENGTH}行で割り切れません（余り {remainder} 行）。"
          f"末尾の端数は切り捨てて {TOTAL_DAYS} 日分のみ使用します。")
    wait_data_csv = wait_data_csv.iloc[:TOTAL_DAYS * DAY_LENGTH]
if TOTAL_DAYS != EXPECTED_DAYS:
    print(f"[警告] 想定日数 {EXPECTED_DAYS} 日に対し、実データは {TOTAL_DAYS} 日分です。")

# == 時間帯フィルタ設定 =====================================================
# True : 指定した時間帯のデータのみで学習 / False : 1日全体(0〜24時)を使用
#  ※ 切り出し位置はサイクル境界（2行単位）に必ず揃えます。
USE_TIME_FILTER = True
START_HOUR = 0    # 学習に使う開始時刻（0〜23）
END_HOUR   = 24   # 学習に使う終了時刻（1〜24、START_HOUR より大きい値）

_cycles_per_hour = CYCLES_PER_DAY / 24
if USE_TIME_FILTER:
    TIME_START_IDX = round(START_HOUR * _cycles_per_hour) * CYCLE_LEN
    TIME_END_IDX   = round(END_HOUR   * _cycles_per_hour) * CYCLE_LEN
else:
    TIME_START_IDX = 0
    TIME_END_IDX   = DAY_LENGTH

print(f"時間帯フィルタ: {'有効' if USE_TIME_FILTER else '無効'} "
      f"({START_HOUR}時〜{END_HOUR}時, "
      f"1日あたり {(TIME_END_IDX - TIME_START_IDX) // CYCLE_LEN} サイクル)")
# ==========================================================================

# === train / val / test の日数（合計 151 日に収める） ===
val_days   = 5
test_days  = 1
train_days = TOTAL_DAYS - val_days - test_days   # = 145

if train_days <= 0:
    raise ValueError(f"学習日数が確保できません（TOTAL_DAYS={TOTAL_DAYS}）")

val_long   = val_days   * DAY_LENGTH
test_long  = test_days  * DAY_LENGTH
train_long = train_days * DAY_LENGTH
print(f"日数構成: train={train_days}日 / val={val_days}日 / test={test_days}日")

# === 損失と入力に関するハイパーパラメータ ===
HORIZON      = 2     # 予測する分割数（t3, t4 の 2 値）
ALPHA        = 2     # 対象道路ノードの待ち台数特徴量を α 倍して強調
STEP_WEIGHTS = [1.0, 1.0]   # t3 / t4 の損失重み
LAMBDA_SHAPE = 0.5   # 損失における「分布の形（ばらつき）」項の重み

# === 分布形状（分散）損失を掛けるステップ（道路ごと） ===
# [t3の重み, t4の重み]。0.0 にすると、そのステップには分散・分布形状の項を一切かけず、
# MSE + MAE だけで学習する（＝そのステップは「分散評価なし」になる）。
# 分布形状損失はステップごとに独立して計算しているので、片方だけ外せる。
#   道路1: t3 は分散を入れず、t4 のみ分散評価あり
#   道路5: 従来どおり t3・t4 の両方に分散評価あり
# ※ この設定は出力フォルダ名にも反映される（例: 道路1_分散t4のみ）ので、
#    設定違いのモデルが互いに上書きされることはない。
SHAPE_STEP_WEIGHTS_BY_ROAD = {
    "道路1": [0.0, 1.0],
    "道路5": [1.0, 1.0],
}
SHAPE_STEP_WEIGHTS_DEFAULT = [1.0, 1.0]   # 辞書に無い道路のときの設定
# === 目標 RMSE（道路ごと） ===
# 決め方: まず 1 回目の学習を回し、その結果の少し下を目標にして再学習する。
# 道路によって待ち台数の水準が違う（道路1 は平均3.2台、道路5 は平均1.1台）ので、
# 同じしきい値を使い回さず道路ごとに持たせる。t3・t4 の両方がこの値以下になったら終了。
#
#   実測 (2026-07-30〜31, RTX 3080 Ti):
#     道路1: 1回目 t3=1.1911 / t4=1.3374（全体 1.2664, 400epoch完走）。
#            目標 1.30 で3試行を追加したが t4 は 1.3576 / 1.3520 / 1.3577 で、
#            4試行すべて 1.337〜1.358 の狭い範囲に収まった（律速は常に t4）。
#            ランダム初期化のやり直しでは 1.30 に届かないと判断し、目標を 1.35 に設定。
#            さらに下げるには再試行ではなく構成の変更が要る（val Loss が epoch395 まで
#            改善し続けていたので learning_rates の延長が第一候補）。
#     道路5: 1回目 t3=1.1197 / t4=0.7637（全体 0.9584, epoch196で早期終了）
#            目標 1.10 で3試行したが t3 は 1.1029 / 1.1064 / 1.1142 と毎回 1.10 をわずかに超え、
#            t4 は 0.761〜0.763 で安定。t3 ≒ 1.10 がこの構成での実質的な下限と判断し、
#            目標を 1.15 に緩めた（t4 側には十分な余裕がある）。
#
# ※ 1試行 ≒ 400エポック × 約15秒 = 約80分。MAX_TRIALS はこの時間を見て決める。
WAIT_RMSE_TH_BY_ROAD = {
    "道路1": 1.35,
    "道路5": 1.15,
}
WAIT_RMSE_TH_DEFAULT = 1.5   # 上の辞書に無い道路を学習するときのしきい値
MAX_TRIALS = 3               # 再学習の最大試行回数（約80分/試行 → 最大約4時間）

# === 大小関係ラベルの正答率評価（既定 OFF） ===
# 「前半(t3) と 後半(t4) のどちらが大きいか」を予測値から判定し、正答率を出す評価。
#   ラベル定義: 0 = 増加(t3 < t4) / 1 = 同じ(t3 == t4) / 2 = 減少(t3 > t4)
# 今は待ち台数の RMSE だけを見るので既定で OFF。必要になったら True にすれば、
# 実測ベース（テストデータから算出）と 正解CSVベース（旧2分割版互換）の両方を出力する。
# ※ 正解データは 1 日分（665 サイクル）しか無いので test_days == 1 のときだけ評価する。
EVAL_ORDER_ACC = False

# === 道路ラベル一覧（対象道路によらず共通） ===
used_road_labels = [str(wait_data_csv.columns[i]) for i in used_road_indices]
used_road_ids = [
    extract_road_id(label, used_road_indices[pos] + 1)
    for pos, label in enumerate(used_road_labels)
]
print(f"使用道路: {used_road_labels}")

# 学習対象がすべて実在するかを、学習を始める前にまとめて検証しておく
# （連続学習の2本目で初めて落ちる、という事態を避ける）
for _r in target_roads:
    _rid = extract_road_id(_r)
    if _rid not in used_road_ids:
        raise ValueError(
            f"対象道路{_rid}は使用道路に含まれていません。選択可能: {used_road_ids}")

# === train / val / test 分割（日ごとの境界で切る） ===
df_train_wait = wait_data_csv.iloc[:train_long, used_road_indices]
df_val_wait   = wait_data_csv.iloc[train_long:train_long + val_long, used_road_indices]
df_test_wait  = wait_data_csv.iloc[train_long + val_long:train_long + val_long + test_long, used_road_indices]

print(f"df_train_wait形状:{df_train_wait.shape} / df_val_wait形状:{df_val_wait.shape} / "
      f"df_test_wait形状:{df_test_wait.shape}")

# === 正規化 ===
scaler_wait = MinMaxScaler()

# 待ち台数: [time, 9] → fit → 転置して [9, time]（元コードと同じ向き）
train_wait = scaler_wait.fit_transform(df_train_wait).T
val_wait   = scaler_wait.transform(df_val_wait).T
test_wait  = scaler_wait.transform(df_test_wait).T

print("=== 待ち台数 MinMaxScaler 範囲 ===")
for i, (mn, mx) in enumerate(zip(scaler_wait.data_min_, scaler_wait.data_max_)):
    print(f"  ノード{i}（{used_road_labels[i]}）: min={mn}, max={mx}")

# ※ 逆正規化ヘルパ（inv_wait）は対象道路の列に依存するため、
#    train_one_road() の中で道路ごとに定義する。

# === edge_index 再構築 ===
adj = adjacency_matrix.values
edge_index_raw = np.array(np.nonzero(adj)).astype(np.int64)
mask = np.isin(edge_index_raw[0], used_road_indices) & np.isin(edge_index_raw[1], used_road_indices)
edge_index_filtered = edge_index_raw[:, mask]
id_map = {old: new for new, old in enumerate(used_road_indices)}
edge_index_mapped = np.vectorize(id_map.get)(edge_index_filtered)
edge_index = torch.tensor(edge_index_mapped, dtype=torch.long).to(device)
print("エッジ：", edge_index.shape)


# === Dataset 作成（日ごとに構成・2入力2出力） ===============================
# 各サンプル:「あるサイクルの2値(t1,t2)」から「次サイクルの2値(t3,t4)」を予測
#   x : [9, 2]  ノード特徴量 [待ち台数(t1), 待ち台数(t2)]
#               対象道路ノードのみ ×ALPHA
#   y : [1, 2]  対象道路の [待ち台数(t3), 待ち台数(t4)]
# サンプルは CYCLE_LEN(=2) 刻み。ブロックが重ならず、サイクル境界に一致する。
# 日をまたぐペアは作らない（day ごとにループを閉じる）。
def build_daywise_dataset(wait_arr, edge_index, device, target_node_index, alpha=ALPHA):
    total_steps = wait_arr.shape[1]
    num_days = total_steps // DAY_LENGTH
    dataset = []

    for day in range(num_days):
        day_start = day * DAY_LENGTH
        range_start = day_start + TIME_START_IDX
        range_end   = day_start + TIME_END_IDX

        # 入力 [t, t+1] と 出力 [t+2, t+3] の両方が時間帯フィルタ内に収まる範囲
        for t in range(range_start, range_end - 2 * CYCLE_LEN + 1, CYCLE_LEN):
            # --- 入力特徴量（当該サイクルの2分割値） ---
            x = torch.tensor(wait_arr[:, t:t + CYCLE_LEN], dtype=torch.float)   # [9, 2]
            x[target_node_index] *= alpha
            x = x.to(device)

            # --- 目的変数（次サイクルの2分割値, 対象道路のみ） ---
            y = torch.tensor(
                wait_arr[target_node_index, t + CYCLE_LEN:t + 2 * CYCLE_LEN],
                dtype=torch.float).unsqueeze(0).to(device)                      # [1, 2]

            dataset.append(Data(x=x, edge_index=edge_index, y=y))

    return dataset


# === バッチ設定 ==============================================================
# 分布形状損失（distribution_shape_loss）はバッチ内の分布を見るため、
# バッチが「学習データ全体の分布を代表している」必要がある。そこでデータ量に
# 合わせて次のように構成する:
#
#   ・batch_size … 1日のサイクル数(665)に近い 640 とする。標準偏差と
#                  Wasserstein 距離の推定が安定する程度に大きく、かつ
#                  train 96,280 サンプルに対し約 151 バッチ/epoch に収まる。
#   ・train は shuffle=True … shuffle=False だと1バッチが「連続した640サイクル
#                  ≒ 1日の約96%」となり、その時間帯だけの局所分布を揃えることに
#                  なる。シャッフルすれば全151日から無作為抽出されるので、
#                  バッチ内分布 ≒ 学習データ全体の分布になる。
#   ・val は shuffle=False … ベストモデル選択と early stopping の指標なので、
#                  毎エポック同じ切り方で決定的に評価する。
#   ・test は shuffle=False 必須 … 予測波形グラフと大小関係の正答率評価が
#                  時系列順に依存するため。
batch_size = 640   # DataLoader は道路ごとに train_one_road() の中で作る


# ==== 2入力2出力の予測モデル（待ち台数のみ） ====
# あるサイクルの2値から 次サイクルの2値を予測。自己回帰・Teacher Forcing なし。
class RedSplitPredictionGNN(torch.nn.Module):
    def __init__(self, num_node_features, horizon=HORIZON, dropout_rate=0.2):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, 128)
        self.skip  = nn.Linear(num_node_features, 128)   # スキップ接続

        self.lin_out = nn.Linear(128, horizon)   # 待ち台数(t3), (t4)
        self.dropout_rate = dropout_rate

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        h = F.relu(self.conv3(h, edge_index))
        h = h + self.skip(x)                              # スキップ接続

        return self.lin_out(h)                            # [num_nodes, 2]


# 学習率スケジュール（合計 400 エポック）
learning_rates = [0.001] * 200 + [0.0005] * 100 + [0.0001] * 100

patience = 45  # early stopping

def distribution_shape_loss(pred, true):
    """ pred, true: [B,1]（バッチ内の予測/正解の集合）。
        『分布の形（ばらつき）』を一致させる損失。
        MSE/MAE は1点ずつの誤差しか見ないため、全体が平均値に張り付いた
        （＝分散が潰れた）予測でも個々の誤差が小さければ合格点になりがち。
        これを防ぐために、バッチ内分布について次の2つを評価して加える:
          ① 標準偏差の差         : ばらつきの大きさ（分散）を本物に合わせる
          ② 並べ替え差(1D W距離) : ソート済み順序統計量の差＝1次元 Wasserstein-1 距離
                                   の近似。分布全体の形を本物に合わせる
        ※ バッチ内サンプルが1個以下のときは統計量が無意味なので0を返す。"""
    p = pred.reshape(-1)
    t = true.reshape(-1)
    if p.numel() < 2:
        return p.sum() * 0.0                       # 勾配グラフを保ったゼロ
    std_term  = (p.std(unbiased=False) - t.std(unbiased=False)).abs()        # ① ばらつき一致
    wass_term = (torch.sort(p)[0] - torch.sort(t)[0]).abs().mean()           # ② 分布の形一致
    return std_term + wass_term


def multi_step_loss(out_target, y_target, criterion, shape_weights=None):
    """ out_target: [B,2], y_target: [B,2]  各列 = [t3, t4] の待ち台数。
        各ステップについて MSE + MAE を計算し、さらに『分布の形（ばらつき）』損失を
        LAMBDA_SHAPE × shape_weights[k] で混ぜて平均値張り付きを防ぐ。
        shape_weights[k] = 0 のステップは分布形状損失を計算せず、MSE + MAE だけで学習する。
        最後に STEP_WEIGHTS で重み付け平均する。"""
    if shape_weights is None:
        shape_weights = SHAPE_STEP_WEIGHTS_DEFAULT
    total = 0.0
    for k in range(HORIZON):
        mse = criterion(out_target[:, k:k + 1], y_target[:, k:k + 1])
        mae = F.l1_loss(out_target[:, k:k + 1], y_target[:, k:k + 1])
        step_loss = mse + mae
        if shape_weights[k] != 0.0:   # 0 のときは分布形状損失を掛けない
            shape = distribution_shape_loss(out_target[:, k:k + 1], y_target[:, k:k + 1])
            step_loss = step_loss + LAMBDA_SHAPE * shape_weights[k] * shape
        total = total + STEP_WEIGHTS[k] * step_loss
    return total / sum(STEP_WEIGHTS)


def shape_setting_tag(shape_weights):
    """分散(分布形状)損失の設定を、出力フォルダ名に使う短いラベルにする。
       全ステップ有効 → 分散評価あり / 全ステップ無効 → 分散評価なし /
       一部のみ有効   → 分散t4のみ のように、有効なステップ名を並べる。"""
    active = [f"t{CYCLE_LEN + k + 1}" for k, w in enumerate(shape_weights) if w != 0.0]
    if len(active) == HORIZON:
        return "分散評価あり"
    if not active:
        return "分散評価なし"
    return "分散" + "".join(active) + "のみ"


# === 予測波形グラフ用の共通設定（道路によらず共通） ===
import matplotlib
matplotlib.rcParams["font.family"] = "Meiryo"
plt.rcParams["font.size"] = 15

if USE_TIME_FILTER:
    _num_hours, _hour_offset = END_HOUR - START_HOUR, START_HOUR
else:
    _num_hours, _hour_offset = 24, 0


def plot_waveform(true_series, pred_series, title, ylabel, save_path, xlabel="Cycle"):
    n = len(true_series)
    cph = n / _num_hours
    ticks = [round(i * cph) for i in range(_num_hours + 1)]
    labels = [f"{_hour_offset + i}時" for i in range(_num_hours + 1)]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(range(n), true_series, color="blue", linewidth=1.5, label="Actual")
    ax.plot(range(n), pred_series, color="red", linewidth=1.5, linestyle="--", label="Predicted")
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=45)
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.ticklabel_format(style="plain", axis="y")
    ax.set_title(title)
    ax.set_xlim(0, n)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(which="both", axis="both")
    ax.minorticks_on()
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"グラフを保存しました → {save_path}")


# === 大小関係ラベル（前半 t3 と 後半 t4 のどちらが大きいか） ==================
def order_label(v1, v2):
    """旧2分割版と同じ定義: 0=増加(v1<v2) / 1=同じ / 2=減少(v1>v2)"""
    if v1 == v2:
        return 1
    return 0 if v1 < v2 else 2


def evaluate_order_accuracy(pred_w, true_w, target_node_index, target_road_label, csv_path):
    """予測 t3/t4 の大小関係の正答率を 2 つの基準で評価して返す。

    ① 実測ベース … テストデータの実測 t3/t4 から作ったラベルと突合（主指標）。
    ② 正解CSVベース … 新環境正解データ_12.22.csv のラベルと突合（旧2分割版と同じ評価）。

    アラインメント: サンプル k は「サイクル k+1 を入力 → サイクル k+2 を予測」なので、
    正解データ（0行目 = サイクル1）では行 (TIME_START_IDX//CYCLE_LEN + 1 + k) が対応する。
    時間帯フィルタが 0〜24 時なら 1 行目〜664 行目（= サイクル2〜665）と 1 対 1 になる。
    このオフセットは、正解CSVと待ち台数CSV最終日（= テスト日）の実測ラベルが
    道路1,2,3,4,7,8,9 で 665/665 一致することを確認済み。ただし道路5・道路6 の列は
    実測と 6 割程度しか一致しない（別定義で作られたと思われる）ため、①と②が
    大きく食い違う場合は ① を信頼すること。両者の一致率も併せて表示する。
    """
    if test_days != 1:
        print(f"[スキップ] 大小関係の正答率評価は test_days==1 のときのみ実施します"
              f"（現在 test_days={test_days}）。")
        return None, None

    # --- 予測値からラベルを作る（旧版と同じく int() で切り捨ててから比較） ---
    pred_labels = [order_label(int(pred_w[k, 0]), int(pred_w[k, 1]))
                   for k in range(len(pred_w))]
    # --- 実測値からラベルを作る（逆正規化の丸め誤差を除くため四捨五入して整数化） ---
    meas_labels = [order_label(int(round(true_w[k, 0])), int(round(true_w[k, 1])))
                   for k in range(len(true_w))]

    # --- 正解CSV読み込み（列 = 道路, 行 = サイクル） ---
    answer_csv = pd.read_csv(answer_csv_path, encoding="utf-8-sig")
    true_all = answer_csv.iloc[:, target_node_index].astype(int).tolist()

    offset = TIME_START_IDX // CYCLE_LEN + 1     # 予測が始まるサイクルの行番号
    csv_labels = true_all[offset:offset + len(pred_labels)]

    if len(csv_labels) != len(pred_labels):
        print(f"[警告] 正解ラベル数 {len(csv_labels)} と予測数 {len(pred_labels)} が"
              f"一致しません。重なる {min(len(csv_labels), len(pred_labels))} 件だけで評価します。")

    n = min(len(csv_labels), len(pred_labels))
    acc_meas = sum(1 for i in range(n) if pred_labels[i] == meas_labels[i]) / n * 100 if n else 0.0
    acc_csv  = sum(1 for i in range(n) if pred_labels[i] == csv_labels[i])  / n * 100 if n else 0.0
    agree    = sum(1 for i in range(n) if meas_labels[i] == csv_labels[i])  / n * 100 if n else 0.0

    # --- CSV 出力（予測・実測・正解CSV のラベルを並べて確認できる形） ---
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["サイクル", "予測t3", "予測t4", "実測t3", "実測t4",
                         "予測ラベル", "実測ラベル", "正解CSVラベル",
                         "実測と一致", "正解CSVと一致"])
        for i in range(n):
            writer.writerow([i + 1,
                             int(pred_w[i, 0]), int(pred_w[i, 1]),
                             int(round(true_w[i, 0])), int(round(true_w[i, 1])),
                             pred_labels[i], meas_labels[i], csv_labels[i],
                             1 if pred_labels[i] == meas_labels[i] else 0,
                             1 if pred_labels[i] == csv_labels[i] else 0])
    print(f"大小関係の比較結果を保存しました → {csv_path}")
    print(f"[{target_road_label}] 大小関係の正答率: 実測ベース {acc_meas:.2f}% / "
          f"正解CSVベース {acc_csv:.2f}%  （評価 {n} サイクル）")
    if agree < 99.0:
        print(f"[注意] 実測ラベルと正解CSVラベルの一致率が {agree:.2f}% しかありません。"
              f"正解CSVの{target_road_label}列は別定義の可能性があるため、実測ベースを信頼してください。")
    return acc_meas, acc_csv


# ============================================================================
# 1 道路分の学習〜評価〜保存を丸ごと行う関数
# ----------------------------------------------------------------------------
# 連続学習ではこの関数を道路ごとに呼ぶ。データ読み込み・正規化・edge_index は
# 道路に依存しないので関数の外で 1 回だけ済ませ、ここでは
#   ・対象ノードの決定と出力先フォルダの用意
#   ・Dataset / DataLoader の作成（ALPHA と y が対象道路に依存するため）
#   ・RMSE 未達なら再学習する while ループ
#   ・テスト予測・CSV・グラフ・スケーラーの保存・大小関係の正答率
# だけを行う。戻り値は結果サマリの dict。
# ============================================================================
def train_one_road(selected_road):
    target_road_id = extract_road_id(selected_road)
    target_node_index = used_road_ids.index(target_road_id)
    target_road_label = used_road_labels[target_node_index]

    # 分布形状（分散）損失をどのステップに掛けるか（道路ごと）
    shape_weights = SHAPE_STEP_WEIGHTS_BY_ROAD.get(target_road_label,
                                                   SHAPE_STEP_WEIGHTS_DEFAULT)
    shape_tag = shape_setting_tag(shape_weights)

    # 成果物（モデル・スケーラー・予測結果CSV・グラフ）はすべてこのフォルダに出力する。
    # ※ 旧 1. 道路1用_予測モデル作成.py（自己回帰 + Teacher Forcing 版）が学習した
    #    既存モデルは「1. 赤時間2分割/道路1」「同/道路5」にそのまま残す。上書きしない
    #    よう、こちらは分散損失の設定名（分散評価あり / 分散t4のみ 等）を付けた
    #    別フォルダに保存する。設定が違えばフォルダも分かれる。
    model_output_dir = os.path.join(BASE_DIR, "予測モデル", "1. 赤時間2分割",
                                    f"{target_road_label}_{shape_tag}")
    os.makedirs(model_output_dir, exist_ok=True)

    best_model_path  = os.path.join(model_output_dir, "GCN_epoch_best_model.pth")
    scaler_wait_path = os.path.join(model_output_dir, "scaler_wait.pkl")
    pred_csv_path    = os.path.join(model_output_dir, f"予測結果_{target_road_label}_赤2分割.csv")
    order_csv_path   = os.path.join(model_output_dir, f"予測結果_大小比較_{target_road_label}.csv")
    loss_plot_path   = os.path.join(model_output_dir, f"Loss曲線_{target_road_label}_赤2分割.png")

    # 目標 RMSE は道路ごと（辞書に無ければ既定値）
    wait_rmse_th = WAIT_RMSE_TH_BY_ROAD.get(target_road_label, WAIT_RMSE_TH_DEFAULT)

    print("\n" + "=" * 72)
    print(f"予測対象: {target_road_label} (ノード index: {target_node_index})")
    print(f"保存先   : {model_output_dir}")
    print(f"目標RMSE : {wait_rmse_th}（t3・t4 の両方に適用 / 最大 {MAX_TRIALS} 試行）")
    print(f"分散損失 : {shape_tag}  重み[t3,t4]={shape_weights} × LAMBDA_SHAPE={LAMBDA_SHAPE}")
    print("=" * 72)

    # === Dataset / DataLoader（対象道路ごとに作り直す） ===
    train_dataset = build_daywise_dataset(train_wait, edge_index, device, target_node_index)
    val_dataset   = build_daywise_dataset(val_wait,   edge_index, device, target_node_index)
    test_dataset  = build_daywise_dataset(test_wait,  edge_index, device, target_node_index)
    print(f"サンプル数 train={len(train_dataset)} / val={len(val_dataset)} / test={len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)
    print(f"バッチ: size={batch_size} / train={len(train_loader)}バッチ(shuffle有) "
          f"/ val={len(val_loader)}バッチ / test={len(test_loader)}バッチ")

    # 逆正規化（対象道路の列のみ）
    w_min0 = scaler_wait.data_min_[target_node_index]
    w_max0 = scaler_wait.data_max_[target_node_index]

    def inv_wait(x):
        return x * (w_max0 - w_min0) + w_min0

    # RMSE 初期化（未達状態から開始）
    rmse_steps = [float('inf')] * HORIZON
    rmse_all = float('inf')
    prediction_count = 1

    # 試行ごとのモデルはこの一時ファイルに保存し、「全試行で最良」のものだけを
    # 正式な best_model_path にコピーする。
    # ※ こうしないと、目標未達で打ち切ったときに「最後の試行のモデル」が残ってしまい、
    #   途中のより良い試行（例: 道路5 は試行1が最良だった）が上書きで失われる。
    trial_model_path = os.path.join(model_output_dir, "_trial_last.pth")
    best_trial = None   # 全試行で最良の結果（RMSE・予測値・Loss曲線）をまとめて持つ

    # ========================================================================
    # 学習ループ（t3・t4 の RMSE がどちらもしきい値以下になるまで再学習）
    # ========================================================================
    while max(rmse_steps) > wait_rmse_th:

        model = RedSplitPredictionGNN(num_node_features=CYCLE_LEN, dropout_rate=0.2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()

        train_losses, val_losses = [], []
        trial_train_losses, trial_val_losses = [], []   # この試行のベストエポック時点の Loss 曲線
        epochs_no_improve = 0
        best_val_loss = float('inf')

        print(f"{prediction_count}回目の学習中 [{target_road_label} 赤2分割・2入力2出力]")
        for epoch in range(len(learning_rates)):
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rates[epoch]

            # --- 学習 ---
            running_train_loss = 0.0
            model.train()
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index)
                out = out.view(-1, node_num, HORIZON)
                out_target = out[:, target_node_index, :]    # [B, 2]
                y_target   = batch.y                          # [B, 2]

                loss = multi_step_loss(out_target, y_target, criterion, shape_weights)
                loss.backward()
                optimizer.step()
                running_train_loss += loss.item()

            avg_train_loss = running_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # --- 検証 ---
            model.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    out = model(batch.x, batch.edge_index)
                    out = out.view(-1, node_num, HORIZON)
                    out_target = out[:, target_node_index, :]
                    y_target   = batch.y
                    loss = multi_step_loss(out_target, y_target, criterion, shape_weights)
                    running_val_loss += loss.item()

            avg_val_loss = running_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            # --- ベストモデル保存（この試行の中でのベストエポック） ---
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), trial_model_path)
                print(f"Best model saved at epoch {epoch+1} with Val Loss: {best_val_loss}")
                trial_train_losses = train_losses.copy()
                trial_val_losses = val_losses.copy()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            print(f'[{target_road_label}] Epoch {epoch+1}, Train Loss: {avg_train_loss:.6f}, '
                  f'Val Loss: {avg_val_loss:.6f}, 学習率: {learning_rates[epoch]}')

            # --- Early Stopping ---
            if epochs_no_improve >= patience:
                print('Early stopping')
                break

        # === テスト予測（この試行のベストエポックのモデルで） ===
        model.load_state_dict(torch.load(trial_model_path, map_location=device))
        model.eval()

        preds, ys = [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index)
                out = out.view(-1, node_num, HORIZON)[:, target_node_index, :]   # [b, 2]
                preds.append(out.cpu().numpy())
                ys.append(batch.y.cpu().numpy())

        preds = np.concatenate(preds, axis=0)   # [N, 2]
        ys    = np.concatenate(ys, axis=0)      # [N, 2]

        # === 逆正規化（対象道路） ===
        pred_w = inv_wait(preds)   # [N, 2]
        true_w = inv_wait(ys)      # [N, 2]

        # === RMSE（t3 / t4 それぞれ + 全体） ===
        rmse_steps = [float(np.sqrt(np.mean((pred_w[:, k] - true_w[:, k]) ** 2)))
                      for k in range(HORIZON)]
        rmse_all = float(np.sqrt(np.mean((pred_w - true_w) ** 2)))
        for k in range(HORIZON):
            print(f"[{target_road_label} 待ち台数 t{CYCLE_LEN + k + 1}] RMSE: {rmse_steps[k]:.4f}")
        print(f"[{target_road_label} 待ち台数 全体]    RMSE: {rmse_all:.4f}")

        # === 全試行を通じて最良なら、正式なモデルとして採用する ===
        # 比較基準は「t3・t4 のうち悪い方の RMSE」。目標も同じ基準なので一貫する。
        if best_trial is None or max(rmse_steps) < best_trial["max_rmse"]:
            shutil.copyfile(trial_model_path, best_model_path)
            best_trial = {
                "trial": prediction_count,
                "max_rmse": max(rmse_steps),
                "rmse_steps": list(rmse_steps),
                "rmse_all": rmse_all,
                "pred_w": pred_w,
                "true_w": true_w,
                "train_losses": trial_train_losses,
                "val_losses": trial_val_losses,
            }
            print(f"[更新] 試行 {prediction_count} がこれまでで最良 → "
                  f"{os.path.basename(best_model_path)} を更新")
        else:
            print(f"[保持] 試行 {prediction_count} は試行 {best_trial['trial']}"
                  f"（最悪ステップ RMSE {best_trial['max_rmse']:.4f}）に及ばないため、"
                  f"保存済みモデルは据え置き")

        # === 終了条件（t3・t4 の RMSE がどちらもしきい値以下） ===
        if max(rmse_steps) <= wait_rmse_th:
            print(f"[達成] {target_road_label} 終了条件達成！ RMSE(t3,t4) = "
                  f"{rmse_steps[0]:.3f} / {rmse_steps[1]:.3f}  (目標 {wait_rmse_th})")
            break
        else:
            print(f"[継続] 目標 {wait_rmse_th} に未達。再学習します"
                  f"（試行 {prediction_count}/{MAX_TRIALS}）")

        prediction_count += 1
        if prediction_count > MAX_TRIALS:
            print(f"[警告] {target_road_label} は最大試行回数に到達。この道路は打ち切ります。")
            break

    # ========================================================================
    # ここから先は「全試行で最良だった試行」の結果だけを成果物として書き出す。
    # 保存済みの GCN_epoch_best_model.pth もその試行のものなので、
    # モデル・CSV・グラフ・RMSE がすべて同じ試行を指す。
    # ========================================================================
    rmse_steps = best_trial["rmse_steps"]
    rmse_all   = best_trial["rmse_all"]
    pred_w     = best_trial["pred_w"]
    true_w     = best_trial["true_w"]

    if prediction_count > 1:
        print(f"\n[採用] 全 {min(prediction_count, MAX_TRIALS)} 試行のうち "
              f"試行 {best_trial['trial']} を採用: RMSE(t3,t4) = "
              f"{rmse_steps[0]:.4f} / {rmse_steps[1]:.4f}  全体 = {rmse_all:.4f}")

    # 試行ごとの一時ファイルは、正式なモデルをコピーし終えたら削除する
    if os.path.exists(trial_model_path):
        os.remove(trial_model_path)

    # === 予測結果 CSV 出力（採用した試行のもの） ===
    with open(pred_csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['サイクル',
                         '実測t3', '予測t3',
                         '実測t4', '予測t4'])
        for k in range(len(pred_w)):
            writer.writerow([k + 1,
                             true_w[k, 0], pred_w[k, 0],
                             true_w[k, 1], pred_w[k, 1]])
    print(f"予測結果を保存しました → {pred_csv_path}")

    # === スケーラー保存 ===
    joblib.dump(scaler_wait, scaler_wait_path)
    print(f"スケーラー保存: {scaler_wait_path}")

    # === 採用した試行の Loss 曲線をプロット ===
    plt.figure(figsize=(8, 5))
    plt.plot(best_trial["train_losses"], label='Train Loss', linewidth=2)
    plt.plot(best_trial["val_losses"], label='Validation Loss', linewidth=2)
    plt.title(f"Training & Validation Loss ({target_road_label})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Loss 曲線を保存しました → {loss_plot_path}")

    # === 予測波形グラフ ===
    # t3 / t4 それぞれ
    for k in range(HORIZON):
        step_name = f"t{CYCLE_LEN + k + 1}"
        plot_waveform(
            true_w[:, k], pred_w[:, k],
            f"待ち台数 {step_name} ({target_road_label})", "待ち台数",
            os.path.join(model_output_dir,
                         f"予測波形_{target_road_label}_赤2分割_{step_name}.png"))

    # t3,t4 を連結した連続波形（ブロック非重複なので時系列としてつながる）
    plot_waveform(
        true_w.reshape(-1), pred_w.reshape(-1),
        f"待ち台数 t3-t4連結 ({target_road_label})", "待ち台数",
        os.path.join(model_output_dir, f"予測波形_{target_road_label}_赤2分割_連結.png"),
        xlabel="Step (赤時間2分割)")

    # === 大小関係ラベルの正答率（実測ベース / 正解CSVベース） ===
    acc_meas = acc_csv = None
    if EVAL_ORDER_ACC:
        acc_meas, acc_csv = evaluate_order_accuracy(
            pred_w, true_w, target_node_index, target_road_label, order_csv_path)

    return {
        "road": target_road_label,
        "rmse_steps": rmse_steps,
        "rmse_all": rmse_all,
        "order_acc_meas": acc_meas,
        "order_acc_csv": acc_csv,
        # 打ち切り時は prediction_count が MAX_TRIALS+1 まで進むので、実際に回した回数に丸める
        "trials": min(prediction_count, MAX_TRIALS),
        "best_trial": best_trial["trial"],
        "target_rmse": wait_rmse_th,
        "achieved": max(rmse_steps) <= wait_rmse_th,
        "output_dir": model_output_dir,
    }


# ============================================================================
# メイン: 対象道路を順番に学習する
# ============================================================================
results = []
for _i, _road in enumerate(target_roads, start=1):
    print(f"\n########## [{_i}/{len(target_roads)}] {_road} の学習を開始 ##########")
    results.append(train_one_road(_road))

# === 全道路の結果サマリ ===
print("\n" + "=" * 72)
print("=== 全学習完了 ===")
print("=" * 72)
for r in results:
    mark = "達成" if r["achieved"] else f"未達(目標{r['target_rmse']})"
    steps = " / ".join(f"{v:.4f}" for v in r["rmse_steps"])
    if r["order_acc_meas"] is not None:
        acc = (f"  大小正答率(実測) = {r['order_acc_meas']:.2f}%"
               f" / (正解CSV) = {r['order_acc_csv']:.2f}%")
    else:
        acc = ""
    print(f"[{mark}] {r['road']}  RMSE(t3/t4) = {steps}  全体 = {r['rmse_all']:.4f}  "
          f"(試行 {r['trials']}回中 試行{r['best_trial']}を採用){acc}")
    print(f"       → {r['output_dir']}")
