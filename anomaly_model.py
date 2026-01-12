import torch
import torch.nn as nn
import pandas as pd
import numpy as np

MODEL_PATH = "model.pth"
CSV_PATH = "data.csv"

# AutoEncoder（構造はそのまま）
class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(3, 2),
            nn.ReLU(),
            nn.Linear(2, 1)
        )
        self.decoder = nn.Sequential(
            nn.Linear(1, 2),
            nn.ReLU(),
            nn.Linear(2, 3)
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out


# -----------------------------
# 正規化のための関数
# -----------------------------
def compute_normalization_params():
    df = pd.read_csv(CSV_PATH)
    data = df.select_dtypes(include=["number"]).values.astype(np.float32)

    mean = data.mean(axis=0)
    std = data.std(axis=0)

    # ゼロ除算防止
    std[std == 0] = 1.0

    return mean, std


def normalize(x, mean, std):
    return (x - mean) / std


def denormalize(x, mean, std):
    return x * std + mean


# -----------------------------
# モデル学習
# -----------------------------
def train_autoencoder_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    data = df.select_dtypes(include=["number"]).values.astype(np.float32)

    mean, std = compute_normalization_params()
    data_norm = normalize(data, mean, std)

    x = torch.tensor(data_norm, dtype=torch.float32)

    model = AutoEncoder()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(300):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, x)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), MODEL_PATH)
    return model, mean, std


# -----------------------------
# モデル + threshold ロード
# -----------------------------
def load_model_and_threshold():
    mean, std = compute_normalization_params()

    model = AutoEncoder()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    df = pd.read_csv(CSV_PATH)
    data = df.select_dtypes(include=["number"]).values.astype(np.float32)

    data_norm = normalize(data, mean, std)
    x = torch.tensor(data_norm, dtype=torch.float32)

    with torch.no_grad():
        reconstructed = model(x)
        reconstructed = reconstructed.numpy()
        x_np = x.numpy()

    # 元スケールに戻す
    x_denorm = denormalize(x_np, mean, std)
    reconstructed_denorm = denormalize(reconstructed, mean, std)

    losses = np.mean((x_denorm - reconstructed_denorm) ** 2, axis=1)

    threshold = float(np.mean(losses) + 3 * np.std(losses))

    return model, mean, std, threshold
