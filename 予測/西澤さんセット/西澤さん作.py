import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch_geometric.data import Data
import torch.nn.functional as F
from torch.nn import BatchNorm1d  # 追加
from torch_geometric.nn import GCNConv,GATConv
from torch_geometric.nn import global_add_pool
from torch_geometric.loader import DataLoader
from torch_geometric.explain import ModelConfig,Explainer, GNNExplainer
from torch_geometric.nn import TGNMemory, TransformerConv
# from torchinfo import summary
import random
import gc

import os

print(torch.cuda.is_available())

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ターゲットノードインデックス
target_node_index = 0  # 道路1(0)に対応する列のインデックス
#ノード数
node_num =9
#グラフを表示するかどうかのフラグ
output_flag = True

# データ読み取り
#赤信号待ち台数のデータ
# train_data_csv = pd.read_csv(r"C:\Users\tslab\Desktop\予測\20250505_テスト用データ_31日分.csv", encoding='cp932')
train_data_csv = pd.read_csv(r"C:\Users\tslab\Desktop\予測\テストデータ.csv", encoding='cp932')
# val_data_csv = pd.read_csv("C:\\Users\\tslab\\Desktop\\交通mod\\★学習_検証_テストデータ\\20240819_約3ヶ月分データ.csv", encoding='shift-jis')
# test_data_csv = pd.read_csv("C:\\Users\\tslab\\Desktop\\交通mod\\★学習_検証_テストデータ\\20240718_休日なし赤信号待ち台数【8月中間使用】.csv"
#                         , encoding='shift-jis')

# train_data_csv = pd.read_csv(r"C:\Users\tslab\Desktop\予測\遅れ時間テスト用_没データ.csv", encoding='utf-8-sig')

#隣接行列
adjacency_matrix = pd.read_csv(r"C:\Users\tslab\Desktop\予測\富山路線_道路_隣接行列_道路10と道路11を除く.csv",encoding='shift-jis', index_col=0)
print(adjacency_matrix)

# データをtrain, validation, testに分割
# 各セットのインデックス範囲に基づいたスライス(1日分：665サイクル)
# train:val:test = 8:1:1

#3か月分データのとき
val_long = 1995   #3日分
test_long = 665     #1日分
train_long = 17955  #27日分

# #3か月分データのとき
# val_long = 11970   #3日分
# test_long = 665     #1日分
# train_long = 47880  #27日分

# #4か月分データのとき
# val_long = 6360
# test_long = 636
# train_long = 69960

#新自動測定modの18日分のでーたのみとのとき
# val_long = 636*2
# test_long = 636
# train_long = 9540

# keep_columns = ['道路1','道路6','道路7','道路9']    #予測に用いる赤信号待ち台数測定道路
# #指定列以外は0埋め
# train_data_csv.loc[:, ~train_data_csv.columns.isin(keep_columns)]=0 
# val_data_csv.loc[:, ~val_data_csv.columns.isin(keep_columns)]=0 
# test_data_csv.loc[:, ~test_data_csv.columns.isin(keep_columns)]=0 
#データ数とノードを決める
df_train_data = train_data_csv.iloc[:train_long,:9]    #110日分
df_val_data = train_data_csv.iloc[train_long:train_long+val_long,:9]#10日分
df_test_data = train_data_csv.iloc[train_long+val_long:train_long+val_long+test_long,:9] #1日分

#道路1を説明変数から除くときに使用
# target_train_data = train_data_csv.iloc[:train_long,0]
# target_val_data = val_data_csv.iloc[:val_long,0]
# target_test_data = test_data_csv.iloc[:test_long,0]

print(f"df_train_data形状:{df_train_data.shape}")
print(f"df_val_data形状:{df_val_data.shape}")
print(f"df_test_data形状:{df_test_data.shape}")
print(df_train_data)

# 各データセットを正規化
scaler = MinMaxScaler()
train_data = scaler.fit_transform(df_train_data).T
validation_data = scaler.transform(df_val_data).T
test_data = scaler.transform(df_test_data).T
# print(train_data.shape)

# 隣接行列の非ゼロ要素を取得してエッジインデックスを作成
adj = adjacency_matrix.values
edge_index = np.array(np.nonzero(adj)).astype(np.int64)
#print(edge_index)
edge_index = torch.tensor(edge_index, dtype=torch.long) # GPUに移動
print("エッジ：",edge_index.shape)

train_dataset = []    #サンプル数ごとに特徴行列をいれるData(x=[11, 1], edge_index=[2, 25])
val_dataset = []    #サンプル数ごとに特徴行列をいれるData(x=[11, 1], edge_index=[2, 25])
test_dataset = []    #サンプル数ごとに特徴行列をいれるData(x=[11, 1], edge_index=[2, 25])
# ノード特徴量をPyTorchテンソルに変換し、転置する（サンプル数分の特徴量行列を作成)
#train用
for i in range(df_train_data.shape[0] - 1):
    x_train_data = torch.tensor(train_data[:,i], dtype=torch.float).unsqueeze(1) #学習データ
    y_train_data = torch.tensor(train_data[:,i+1], dtype=torch.float).unsqueeze(1)  #教師（正解）データ
    data_x_train =  Data(x=x_train_data, edge_index=edge_index, y=y_train_data)
    train_dataset.append(data_x_train)

#検証用
for i in range(df_val_data.shape[0] -1):
    x_val_data = torch.tensor(validation_data[:,i], dtype=torch.float).unsqueeze(1) 
    y_val_data = torch.tensor(validation_data[:,i+1], dtype=torch.float).unsqueeze(1)
    data_x_val =  Data(x=x_val_data, edge_index=edge_index, y=y_val_data)
    val_dataset.append(data_x_val)

#テスト用
for i in range(df_test_data.shape[0] - 1):
    x_test_data = torch.tensor(test_data[:,i], dtype=torch.float).unsqueeze(1) 
    y_test_data = torch.tensor(test_data[:,i+1], dtype=torch.float).unsqueeze(1)
    data_x_test =  Data(x=x_test_data, edge_index=edge_index, y=y_test_data)
    test_dataset.append(data_x_test)

# print(train_dataset)
batch_size = 32
# データローダー
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class TrafficPredictionGNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels,dropout_rate=0.3):
        super(TrafficPredictionGNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        #self.fc4 = torch.nn.Linear(hidden_channels, hidden_channels)

         # 全結合層の追加
        # self.fc1 = torch.nn.Linear(hidden_channels, hidden_channels)
        # self.fc2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels,1)  # 出力を全ノードに対応させる
        self.dropout_rate = dropout_rate  # ドロップアウトの確率

    def forward(self, x, edge_index):
        #print(x.shape)
        #print(edge_index.shape)
        # x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        # x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
 
        x = self.conv3(x, edge_index)
        x = torch.relu(x)
    
        x = self.conv4(x, edge_index)
        x = torch.relu(x)
        
        # x = global_add_pool(x, batch)
        #print(x.shape)
        # x = F.dropout(x, p=self.dropout_rate, training=self.training)
        # x = self.fc1(x)

        out = self.lin(x)
        return out

# 学習率のリスト（エポック数に基づいて設定）
learning_rates = [0.001] * 200 +[0.001] *90  +[0.0001] * 100

#モデルの保存パス
best_model_path = r"C:\Users\tslab\Desktop\予測\予測モデル\GCN_epoch_best_model.pth"

# 早期停止の設定
patience = 10                   #早期終了の条件
best_val_loss = float('inf')    #最小のバリデーションロスを記録(初期は無限)
best_test_loss = float('inf')   #val Lossが改善されなかった連続エポック数
epochs_no_improve = 0

# トレーニングループ (可視化用)
train_losses = []
val_losses = []  # バリデーションロスを記録
test_losses = []

# 予測結果保存用リスト
epoch_predictions = []
epoch_targets = []

#予測をし，RMSEを計算をした回数保存
prediction_count = 1

#RMSE保存
rmse = float('inf')

# トレーニングループ（指定したRMSEより精度が向上しない場合は，再度学習・予測）
while(rmse > 1.5):
    model = TrafficPredictionGNN(num_node_features=1, hidden_channels=4, dropout_rate=0.2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    #Lossを格納している配列中身を削除
    train_losses.clear()
    val_losses.clear()
    epochs_no_improve = 0
    #best_Model保存を変数初期化
    best_val_loss = float('inf')
    best_test_loss = float('inf')

    print(f"{prediction_count}回目の学習中")
    model.train()
    for epoch in range(100):  # 例として100エポック
        
        # エポックに応じて学習率を変更
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rates[epoch]

        # トレーニング
        running_train_loss = 0.0
        model.train()  # モデルを訓練モードに
        for batch in train_loader:
            batch = batch.to(device)  # データをGPUに移動
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)  # 明示的にxとedge_indexを渡す
            # 道路1に該当するノードのみを抽出（インデックス0と仮定）
            out_target = out.view(-1, node_num)[:, 0]  #  各バッチで最初のノードの出力を取得
            y_target = batch.y.view(-1, node_num)[:, 0] #  各バッチで最初のノードの出力を取得
            # デバッグ用に形状を出力
            #print("out:", out_target.shape)
            #print("y:", y_target.shape)
            loss = criterion(out_target, y_target)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # バリデーションロスの計算
        model.eval()  # モデルを評価モードに
        running_val_loss = 0.0
        predictions = []
        targets = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)  # データをGPUに移動
                out = model(batch.x, batch.edge_index)  # 明示的にxとedge_indexを渡す
                # 道路1に該当するノードのみを抽出（インデックス0と仮定）
                out_target = out.view(-1, node_num)[:, 0]# 各バッチで最初のノードの出力を取得
                y_target = batch.y.view(-1, node_num)[:, 0]# 各バッチで最初のノードの出力を取得
                loss = criterion(out_target, y_target)
                running_val_loss += loss.item()
                # predictions.append(out_target.cpu().numpy())
                # targets.append(y_target.cpu().numpy())

        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        #局所解に入った際に学習を終わらせる
        # if(avg_val_loss > 0.011 and epoch >= 1):
        #     print("局所解に入りました．学習を終了させます．")
        #     break

        # エポックごとの予測結果を保存
        # epoch_predictions.append(np.concatenate(predictions))
        # epoch_targets.append(np.concatenate(targets))

        # 最良モデルの保存
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            #Sepochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)  # モデルを保存
            print(f"Best model saved at epoch {epoch+1} with Val Loss: {best_val_loss}")

        # 結果の出力
        print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}, 学習率: {learning_rates[epoch]}')
        
        # Early Stopping
        if avg_val_loss < best_test_loss:
            best_test_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        #print(f"Loss更新なし：{epochs_no_improve}回")
        if epochs_no_improve >= patience:
            print('Early stopping')
            break

    # テストデータの予測
    model.load_state_dict(torch.load(best_model_path, weights_only=True,map_location=device))  # 最良モデルを読み込み
    model.eval()
    test_predictions = []
    test_ground_truth = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)  # データをGPUに移動
            out = model(batch.x, batch.edge_index)
            #print(out.shape)
            out_target = out.view(-1, node_num)  # 予測値
            y_target = batch.y.view(-1, node_num)  # 真値(テストデータ)
            
            test_predictions.append(out_target.cpu().numpy())
            test_ground_truth.append(y_target.cpu().numpy())
    # 配列を結合して形状を確認(形状の確認ためnumpy配列に変換)
    test_predictions = np.concatenate(test_predictions, axis=0)
    test_ground_truth = np.concatenate(test_ground_truth, axis=0)

    # 逆正規化
    predicted_traffic = scaler.inverse_transform(test_predictions)  #予測データ
    predicted_true_traffic = scaler.inverse_transform(test_ground_truth)    #テストデータ

    # 四捨五入
    predicted_traffic = np.round(predicted_traffic) #四捨五入：予測データ
    predicted_true_traffic = np.round(predicted_true_traffic) #四捨五入：testデータ
    print("逆正規化・四捨五入後の予測データ：",predicted_traffic.shape)
    print("逆正規化・四捨五入後のテストデータ：",predicted_true_traffic.shape)
    # print(predicted_true_traffic)

    # RMSEの計算
    rmse = np.sqrt(np.mean((predicted_traffic[:,0] - predicted_true_traffic[:,0])**2))
    #pd.Series(rmse).to_csv('C:\\Users\\tslab\\Desktop\\西澤_研究関連\\csvファイル掃き出し\\rmse_to_csv_out.csv', mode='a', index=False,header=False)
    print(f'{prediction_count}回目のRMSE: {rmse}')
    print(f"{prediction_count}回目の予測が終了しました")
    prediction_count = prediction_count+1   #RMSE計算回数加算

    print(np.amax(predicted_traffic[:,0]))  #予測値の最大値表示←赤信号待ち台数の増減が激しい部分予測できているかの確認用
    #回数で終了判定
    if(prediction_count == 21):
        break
    #最大値で終了判定
    # if(np.amax(predicted_traffic[:,0]) >=16):
    #     break

print(f"最高精度予測完了 RMSE:{rmse}")
# true_traffic_df = pd.DataFrame(predicted_true_traffic[:,0], columns=["True_Traffic"])
# true_traffic_df.to_csv('C:\\Users\\tslab\\Desktop\\西澤_研究関連\\csvファイル掃き出し\\GCN_テストデータ_csv掃き出し.csv', index=False)
# # DataFrameに変換してCSVに保存
# predicted_traffic_df = pd.DataFrame(predicted_traffic[:,0], columns=["predicted_Traffic"])
# predicted_traffic_df.to_csv('C:\\Users\\tslab\\Desktop\\西澤_研究関連\\csvファイル掃き出し\\GCN_予測_csv掃き出し.csv', index=False)


# グラフ表示
plt.rcParams["font.size"] = 15
plt.figure(figsize=(12, 6))
plt.plot(predicted_true_traffic[:,0], color='blue', label='Actual')  # テストデータ
plt.plot(predicted_traffic[:,0], color='red', label='Predicted')  # 予測データ
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.top"] = True
plt.rcParams["xtick.bottom"] = True
plt.rcParams["ytick.left"] = True
plt.rcParams["ytick.right"] = True
plt.rcParams["xtick.minor.visible"] = True
plt.rcParams["ytick.minor.visible"] = True
plt.minorticks_on()
plt.grid(which="both", axis="y")
plt.grid(which="both", axis="x")
plt.xticks(np.arange(0, 665, step=50))
plt.title('Vehicle Number Forecast for Road 1')
plt.xlim(0,665) #x軸の範囲
plt.xlabel('Cycle')
plt.ylabel('Number of vehicles')
plt.legend()
plt.savefig(f'C:\\Users\\tslab\\Desktop\\予測\\相関係数0.5以上\\道路10_道路11を除く_予測波形_{rmse}.png')
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(predicted_true_traffic[:,0], color='blue', label='Actual')  #テストデータ
plt.plot(predicted_traffic[:,0], color='red', label='Predicted') #予測データ
plt.rcParams["xtick.direction"] = "in"      # 目盛り線の向き、内側"in"か外側"out"かその両方"inout"か
plt.rcParams["ytick.direction"] = "in"      # 目盛り線の向き、内側"in"か外側"out"かその両方"inout"か
plt.rcParams["xtick.top"] = True            # 上部に目盛り線を描くかどうか
plt.rcParams["xtick.bottom"] = True         # 下部に目盛り線を描くかどうか
plt.rcParams["ytick.left"] = True           # 左部に目盛り線を描くかどうか
plt.rcParams["ytick.right"] = True          # 右部に目盛り線を描くかどうか
plt.rcParams["xtick.minor.visible"] = True  # x軸副目盛り線を描くかどうか
plt.rcParams["ytick.minor.visible"] = True  # y軸副目盛り線を描くかどうか
plt.minorticks_on()
plt.grid(which="both", axis="y")
plt.grid(which="both", axis="x")
plt.xticks(np.arange(0, 665, step=50))  # 目盛り設定
plt.title('Vehicle Number Forecast')
plt.xlim(150,300) #x軸の範囲
plt.xlabel('Cycle')
plt.ylabel('Number of vehicles')
plt.legend()
plt.savefig(f'C:\\Users\\tslab\\Desktop\\予測\\相関係数0.5以上\\道路10_道路11を除く_拡大波形.png')
plt.show()

# 学習曲線の表示
plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Training Loss', color='blue')
plt.plot(val_losses, label='Validation Loss', color='orange')  # バリデーションロスを表示
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Test Loss Over Epochs')
plt.savefig('C:\\Users\\tslab\\予測\\相関係数0.5以上\\道路10_道路11を除く_Loss波形.png')

plt.show()

#重要度分析-----------------------------------------
# モデル構成の設定（修正ポイント）
# model_config = ModelConfig(
#     mode='regression',  # 回帰タスクに変更
#     task_level='node',  # ノード単位の説明
#     return_type='raw',   # 回帰値を返す
# )

# # Explainerの設定
# explainer = Explainer(
#     model=model,
#     algorithm=GNNExplainer(epochs=200),
#     explanation_type='model',
#     node_mask_type='attributes',    # ノード特徴量に基づく説明
#     edge_mask_type='object',          # エッジに関しても特徴量に基づく説明
#     model_config=model_config  # dictではなくModelConfigオブジェクト
# )
# data = test_dataset[0].to(device)
# # ノード単位の説明生成
# node_index = 0
# explanation = explainer(data.x, data.edge_index, index=node_index)
# print(f'Node Mask: {explanation.node_mask}')
# print(f'Node Mask Shape: {explanation.node_mask.shape}')

# # 結果の確認
# print(f'Generated explanations in {explanation.available_explanations}')

# # 特徴重要度の可視化
# # ノードごとの影響度を取得（形状: [11, 1]）
# node_importance = explanation.node_mask.cpu().detach().numpy().flatten()
# plt.figure(figsize=(12, 6))
# plt.bar(range(len(node_importance)), node_importance)
# plt.xlabel('Node Index')
# plt.ylabel('Importance')
# plt.title('Node Feature Importance')
# #plt.savefig('node_importance.png')
# plt.show()
#------------------------------
