import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('Qt5Agg')  # Use Qt for interactive plotting
import matplotlib.pyplot as plt

# Configs
WINDOW_SIZE = 25
BATCH_SIZE = 64
EPOCHS = 100
PATIENCE = 100
LR = 0.001
NUM_WORKERS = min(32, os.cpu_count()) - 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data paths
DATA_PATH = Path("datasets/CMaps/")
TRAIN_FILE = DATA_PATH / "train_FD001.txt"
TEST_FILE = DATA_PATH / "test_FD001.txt"
RUL_FILE = DATA_PATH / "RUL_FD001.txt"

def load_dataset(path):
    col_names = ["unit", "time", "os1", "os2", "os3"] + [f"s_{i}" for i in range(1, 22)]
    return pd.read_csv(path, sep=" ", header=None, names=col_names, usecols=range(26))

def add_rul(df):
    max_cycle = df.groupby("unit")["time"].max().reset_index()
    max_cycle.columns = ["unit", "max"]
    df = df.merge(max_cycle, on="unit", how="left")
    df["RUL"] = df["max"] - df["time"]
    df.drop(columns=["max"], inplace=True)
    return df

# Drop unused columns
drop_cols = ["unit", "time", "s_1", "s_5", "s_6", "s_10", "s_16", "s_18", "s_19"]
slope_sensor_indices = [2, 3, 4, 7, 8, 11, 12, 13, 15, 17, 20, 21]
slope_sensor_names = [f"s_{i}" for i in slope_sensor_indices]
extra_features = [f"slope_s_{i}" for i in slope_sensor_indices]

# Load and prepare training data
train_df = load_dataset(TRAIN_FILE)
train_df = add_rul(train_df)
unit_column = train_df["unit"].copy()
train_df = train_df.drop(columns=[col for col in drop_cols if col != "unit"])
feature_cols = [col for col in train_df.columns if col not in ["RUL", "unit"]]
scaler = StandardScaler()
train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
train_df["unit"] = unit_column

def generate_sequences(df, window_size):
    sequences, labels = [], []
    for unit in df["unit"].unique():
        unit_df = df[df["unit"] == unit].reset_index(drop=True)
        for i in range(len(unit_df) - window_size + 1):
            seq_df = unit_df.iloc[i:i + window_size]
            seq_data = seq_df[feature_cols].values
            x_time = np.arange(window_size).reshape(-1, 1)
            slopes = []
            for sensor in slope_sensor_names:
                y = seq_df[sensor].values.reshape(-1, 1)
                model = LinearRegression().fit(x_time, y)
                slopes.append(model.coef_[0][0])
            extended_seq = np.hstack((seq_data, np.tile(slopes, (window_size, 1))))
            sequences.append(extended_seq)
            labels.append(seq_df["RUL"].values[-1])
    return np.array(sequences), np.array(labels)

X_train, y_train = generate_sequences(train_df, WINDOW_SIZE)

# Prepare test data
test_df = load_dataset(TEST_FILE)
rul_truth = pd.read_csv(RUL_FILE, header=None)[0]
unit_column = test_df["unit"].copy()
test_df = test_df.drop(columns=[col for col in drop_cols if col != "unit"])
test_df[feature_cols] = scaler.transform(test_df[feature_cols])
test_df["unit"] = unit_column

X_test = []
for unit in test_df["unit"].unique():
    unit_df = test_df[test_df["unit"] == unit].reset_index(drop=True)
    if len(unit_df) >= WINDOW_SIZE:
        seq_df = unit_df.iloc[-WINDOW_SIZE:]
    else:
        pad = pd.DataFrame(0, index=np.arange(WINDOW_SIZE - len(unit_df)), columns=feature_cols)
        seq_df = pd.concat([pad, unit_df[feature_cols]], ignore_index=True)
    x_time = np.arange(WINDOW_SIZE).reshape(-1, 1)
    slopes = []
    for sensor in slope_sensor_names:
        y = seq_df[sensor].values.reshape(-1, 1)
        model = LinearRegression().fit(x_time, y)
        slopes.append(model.coef_[0][0])
    extended_seq = np.hstack((seq_df[feature_cols].values, np.tile(slopes, (WINDOW_SIZE, 1))))
    X_test.append(extended_seq)

X_test = np.array(X_test)
y_test = rul_truth.values
all_feature_cols = feature_cols + extra_features

# Dataset
class RULSequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(RULSequenceDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
test_loader = DataLoader(RULSequenceDataset(X_test, y_test), batch_size=1, num_workers=NUM_WORKERS)

# Model
class RULLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=4):
        super(RULLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])
    
class ImprovedRULLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, bidirectional=True, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )

        self.norm = nn.LayerNorm(hidden_size * self.num_directions)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)                        # shape: (B, T, D)
        last_hidden = out[:, -1, :]                  # use final time step
        normed = self.norm(last_hidden)              # normalize before FC
        return self.fc(normed)
    
class BalancedAsymmetricLoss(nn.Module):
    def __init__(self, over_weight=2.0, under_weight=1.0, scale=5.0, cap_limit=150.0, cap_weight=0.1):
        super().__init__()
        self.over_weight = over_weight
        self.under_weight = under_weight
        self.scale = scale
        self.cap_limit = cap_limit
        self.cap_weight = cap_weight

    def forward(self, pred, target):
        diff = pred - target

        # Main asymmetric loss
        base_loss = torch.where(
            diff > 0,
            self.over_weight * (torch.exp(diff / self.scale) - 1),
            self.under_weight * diff ** 2
        )

        # Soft cap penalty
        cap_penalty = torch.relu(pred - self.cap_limit) ** 2

        # Combine losses
        loss = base_loss + self.cap_weight * cap_penalty
        return loss.mean()



# model = RULLSTM(input_size=len(all_feature_cols)).to(device)
model = ImprovedRULLSTM(input_size=len(all_feature_cols)).to(device)
criterion = BalancedAsymmetricLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)

# Training
best_loss = float("inf")
trigger_times = 0
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    scheduler.step(avg_loss)
    if avg_loss < best_loss - 1e-4:
        best_loss = avg_loss
        trigger_times = 0
    else:
        trigger_times += 1
        if trigger_times >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

# Evaluation
model.eval()
preds, targets = [], []
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        pred = model(x).cpu().item()
        pred = min(pred, 150)
        preds.append(pred)
        targets.append(y.item())

mae = mean_absolute_error(targets, preds)
rmse = np.sqrt(mean_squared_error(targets, preds))
r2 = r2_score(targets, preds)

print(f"\nMAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²:   {r2:.4f}")

# Bar chart
indices = np.arange(len(targets))
plt.figure(figsize=(12, 6))
bar_width = 0.4
plt.bar(indices - bar_width/2, targets, width=bar_width, color='orange', label='True RUL')
plt.bar(indices + bar_width/2, preds, width=bar_width, color='blue', label='Predicted RUL')
plt.xlabel("Engine")
plt.ylabel("RUL")
plt.title("True vs Predicted RUL")
plt.legend()
plt.grid(axis='y')
plt.tight_layout()
plt.show()

