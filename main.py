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

import matplotlib
matplotlib.use('Qt5Agg')  # Use Qt for interactive plotting
import matplotlib.pyplot as plt

# Configs
WINDOW_SIZE = 30
BATCH_SIZE = 64
EPOCHS = 100
PATIENCE = 1000
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

# Load data
train_df = load_dataset(TRAIN_FILE)
test_df = load_dataset(TEST_FILE)
rul_truth = pd.read_csv(RUL_FILE, header=None)[0]

# Add RUL column
def add_rul(df):
    max_cycle = df.groupby("unit")["time"].max().reset_index()
    max_cycle.columns = ["unit", "max"]
    df = df.merge(max_cycle, on="unit", how="left")
    df["RUL"] = df["max"] - df["time"]
    df.drop(columns=["max"], inplace=True)
    return df

train_df = add_rul(train_df)

# Drop unused columns
drop_cols = ["unit", "time", "s_1", "s_5", "s_6", "s_10", "s_16", "s_18", "s_19"]
train_df = train_df.drop(columns=drop_cols)
feature_cols = [col for col in train_df.columns if col != "RUL"]

# Scale
scaler = StandardScaler()
train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])

# Prepare sequences
def generate_sequences(df, window_size):
    sequences = []
    labels = []
    for unit in df["unit"].unique():
        unit_df = df[df["unit"] == unit]
        unit_df = unit_df.reset_index(drop=True)
        for i in range(len(unit_df) - window_size + 1):
            seq = unit_df.iloc[i:i+window_size]
            sequences.append(seq[feature_cols].values)
            labels.append(seq["RUL"].values[-1])
    return np.array(sequences), np.array(labels)

train_df_seq = load_dataset(TRAIN_FILE)
train_df_seq = add_rul(train_df_seq)
unit_column = train_df_seq["unit"].copy()
train_df_seq = train_df_seq.drop(columns=[col for col in drop_cols if col != "unit"])
train_df_seq[feature_cols] = scaler.transform(train_df_seq[feature_cols])
train_df_seq["unit"] = unit_column
X_train, y_train = generate_sequences(train_df_seq, WINDOW_SIZE)

# Test set (use last WINDOW_SIZE cycles per engine)
# Load test data
test_df_seq = load_dataset(TEST_FILE)

# Preserve unit info
unit_column = test_df_seq["unit"].copy()

# Drop only the necessary columns (but keep 'unit')
test_df_seq = test_df_seq.drop(columns=[col for col in drop_cols if col != "unit"])

# Scale features
test_df_seq[feature_cols] = scaler.transform(test_df_seq[feature_cols])

# Restore unit column (for sequence grouping)
test_df_seq["unit"] = unit_column

X_test = []
for unit in test_df_seq["unit"].unique():
    unit_df = test_df_seq[test_df_seq["unit"] == unit]
    if len(unit_df) >= WINDOW_SIZE:
        seq = unit_df.iloc[-WINDOW_SIZE:][feature_cols].values
    else:
        pad = np.zeros((WINDOW_SIZE - len(unit_df), len(feature_cols)))
        seq = np.vstack((pad, unit_df[feature_cols].values))
    X_test.append(seq)

X_test = np.array(X_test)
y_test = rul_truth.values

# Dataset and DataLoader
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

# LSTM Model
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
        out = out[:, -1, :]  # use last time step
        return self.fc(out)

model = RULLSTM(input_size=len(feature_cols)).to(device)
criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

# Training loop with early stopping
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
        preds.append(pred)
        targets.append(y.item())

mae = mean_absolute_error(targets, preds)
rmse = np.sqrt(mean_squared_error(targets, preds))
r2 = r2_score(targets, preds)

print(f"\nMAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²:   {r2:.4f}")

indices = np.arange(len(targets))

plt.figure(figsize=(12, 6))
bar_width = 0.4

# Bar charts for true and predicted RUL
plt.bar(indices - bar_width/2, targets, width=bar_width, color='orange', label='True RUL')
plt.bar(indices + bar_width/2, preds, width=bar_width, color='blue', label='Predicted RUL')

plt.xlabel("Engine")
plt.ylabel("RUL")
plt.title("True vs Predicted RUL")
plt.legend()
plt.grid(axis='y')
plt.tight_layout()
plt.show()
