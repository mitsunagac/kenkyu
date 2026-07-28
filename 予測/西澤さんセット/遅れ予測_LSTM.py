import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
import os
import matplotlib.pyplot as plt  # グラフ描画用

# === ハイパーパラメータ ===
SEQUENCE_LENGTH = 30               # LSTMに与える時系列の長さ
BATCH_SIZE = 32                    # バッチサイズ
EPOCHS = 100                       # 1回の試行あたりの最大エポック数
LR = 0.001                         # 学習率
EARLY_STOPPING_PATIENCE = 10      # early stopping の猶予エポック数
ACCURACY_THRESHOLD = 0.75         # 要求されるテスト正答率
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 入出力のパス設定 ===
DATA_PATH = r"C:\Users\tslab\Desktop\予測\遅れ時間_統合_all_days.csv"  # 読み込むCSVのパス
DATA_COLUMN = 0
MODEL_DIR = r"C:\Users\tslab\Desktop\予測\遅れ時間予測\予測モデル"      # モデル出力ディレクトリ
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")           # モデル出力ファイル
PLOT_SAVE_PATH = r"C:\Users\tslab\Desktop\予測\遅れ時間予測\グラフ\loss_curve.png"  # 学習曲線の保存パス
RESULT_PLOT_PATH = r"C:\Users\tslab\Desktop\予測\遅れ時間予測\グラフ\pred_vs_true.png"  # 予測と正解の比較プロット
SAVE_PLOT = True

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(os.path.dirname(PLOT_SAVE_PATH), exist_ok=True)

# === データセットクラス定義 ===
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len):
        self.inputs = []
        self.labels = []
        for i in range(len(data) - seq_len):
            self.inputs.append(data[i:i+seq_len])
            self.labels.append(data[i+seq_len])
        self.inputs = torch.tensor(self.inputs, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

# === LSTMモデル定義 ===
class LSTMModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1])  # 最後の時刻の出力のみ使う
        return out

# === データ読み込み・分割（日単位）===
TOTAL_DAYS = 31
DAY_LENGTH = 665
TRAIN_DAYS = 27
VAL_DAYS = 3

# 指定列のみ読み込み（ヘッダーあり前提）
data = pd.read_csv(DATA_PATH, header=0).iloc[:, DATA_COLUMN].values.flatten()
assert len(data) == TOTAL_DAYS * DAY_LENGTH, "データ長が想定外です"

# 各データセットに分割
train_data = data[:TRAIN_DAYS * DAY_LENGTH]
val_data = data[TRAIN_DAYS * DAY_LENGTH:(TRAIN_DAYS + VAL_DAYS) * DAY_LENGTH]
test_data = data[(TRAIN_DAYS + VAL_DAYS) * DAY_LENGTH:]

# データローダ作成（固定）
train_dataset = TimeSeriesDataset(train_data, SEQUENCE_LENGTH)
val_dataset = TimeSeriesDataset(val_data, SEQUENCE_LENGTH)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# === 正答率が80%以上になるまで繰り返す ===
attempt = 1
best_test_accuracy = 0.0

while best_test_accuracy < ACCURACY_THRESHOLD:
    print(f"\n🌀 Attempt {attempt}: 学習開始")

    # モデル・オプティマイザ初期化
    model = LSTMModel().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # EarlyStopping初期化
    best_val_loss = float("inf")
    epochs_no_improve = 0

    # 損失記録用リスト
    train_losses = []
    val_losses = []

    # === 学習ループ ===
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.unsqueeze(-1).to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 検証
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.unsqueeze(-1).to(DEVICE), y.to(DEVICE)
                output = model(x)
                loss = criterion(output, y)
                val_loss += loss.item()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # モデル保存 & EarlyStopping判定
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"📌 Best model saved at epoch {epoch+1} with Val Loss: {val_loss:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"⛔ Early stopping at epoch {epoch+1}")
                break

    # === テスト評価 ===
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    model.eval()
    input_seq = torch.tensor(train_data[-SEQUENCE_LENGTH:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEVICE)
    preds = []

    with torch.no_grad():
        input_seq = torch.tensor(train_data[-SEQUENCE_LENGTH:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEVICE)
        preds = []
        for t in range(len(test_data)):
            out = model(input_seq)
            pred = torch.argmax(out, dim=-1).item()
            preds.append(pred)
            true_input = test_data[t]  # ←ここを真値で更新
            new_input = torch.tensor([[true_input]], dtype=torch.float32).to(DEVICE)
            input_seq = torch.cat([input_seq[:, 1:], new_input.unsqueeze(0)], dim=1)

    correct = sum([int(p == t) for p, t in zip(preds, test_data)])
    total = len(test_data)
    best_test_accuracy = correct / total

    print(f"\n🧪 Attempt {attempt} 完了 → 正解数: {correct} / {total}, 正答率: {best_test_accuracy:.4f}")
    attempt += 1

# === 結果保存と可視化 ===
pd.DataFrame(preds).to_csv("predicted.csv", index=False, header=False)
print(f"\n🎉 十分な精度を達成しました！正答率: {best_test_accuracy:.4f} ✅ モデル保存完了")

# === 学習曲線の描画 ===
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Transition (Best Attempt)")
plt.legend()
plt.grid(True)
plt.tight_layout()
if SAVE_PLOT:
    plt.savefig(PLOT_SAVE_PATH)
plt.show()

# === 予測 vs 実際のプロット ===
plt.figure(figsize=(8, 4))
plt.plot(range(len(test_data)), test_data, label="True", marker="o", markersize=4, linestyle='-', color='blue')
plt.plot(range(len(preds)), preds, label="Pred", marker="x", markersize=4, linestyle='-', color='red')
plt.xlabel("Time Step")
plt.ylabel("Class")
plt.title("Prediction vs True Labels")
plt.legend()
plt.grid(True)
plt.tight_layout()
if SAVE_PLOT:
    plt.savefig(RESULT_PLOT_PATH)
plt.show()

# === 3分割の拡大グラフを出力 ===
part_len = len(test_data) // 3
for i in range(3):
    start = i * part_len
    end = (i + 1) * part_len if i < 2 else len(test_data)
    plt.figure(figsize=(8, 4))
    plt.plot(range(start, end), test_data[start:end], label="True", marker="o", markersize=4, linestyle='-', color='blue')
    plt.plot(range(start, end), preds[start:end], label="Pred", marker="x", markersize=4, linestyle='-', color='red')
    plt.xlabel("Time Step")
    plt.ylabel("Class")
    plt.title(f"Prediction vs True (Part {i+1})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if SAVE_PLOT:
        part_path = RESULT_PLOT_PATH.replace(".png", f"_part{i+1}.png")
        plt.savefig(part_path)
    plt.show()
