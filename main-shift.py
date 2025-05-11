import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import joblib

# --- Load dataset ---
df = pd.read_csv("datasets/shift-data/train_FD001_with_humans.csv")

# --- Feature Engineering ---
drop_columns = ["op_setting_1", "op_setting_2", "op_setting_3"] + [f"sensor_{i}" for i in range(1, 22)]
X = df.drop(columns=drop_columns + ["time_cycles", "unit_number"])
y = df["time_cycles"]

# Normalize numeric columns
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
for col in numerical_cols:
    mean = X[col].mean()
    std = X[col].std()
    X[f"{col}_norm"] = (X[col] - mean) / std if std != 0 else 0.0
X.drop(columns=numerical_cols, inplace=True)

# Encode categorical variables
X = pd.get_dummies(X, columns=["shift_type", "experience_level", "gender"], drop_first=True)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "mlp_shift_scaler.pkl")

# --- Train/Val Split ---
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- Dataset ---
class ShiftDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32).unsqueeze(-1)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

train_loader = DataLoader(ShiftDataset(X_train, y_train), batch_size=64, shuffle=True)
val_loader = DataLoader(ShiftDataset(X_val, y_val), batch_size=64)

# --- MLP Model ---
class ImprovedMLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.net(x)

# --- Training ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImprovedMLP(X.shape[1]).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

best_loss = float("inf")
patience, wait = 40, 0

for epoch in range(300):
    model.train()
    train_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            val_loss += criterion(model(xb), yb).item()
    val_loss /= len(val_loader)
    scheduler.step(val_loss)

    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.2f}, Val Loss: {val_loss:.2f}")

    if val_loss < best_loss - 1e-4:
        best_loss = val_loss
        wait = 0
        torch.save(model.state_dict(), "mlp_shift_model.pth")
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping")
            break

# --- Evaluation ---
model.eval()
preds, targets = [], []
with torch.no_grad():
    for xb, yb in val_loader:
        xb = xb.to(device)
        pred = model(xb).cpu().numpy().flatten()
        preds.extend(pred)
        targets.extend(yb.numpy().flatten())

mae = mean_absolute_error(targets, preds)
rmse = np.sqrt(mean_squared_error(targets, preds))
r2 = r2_score(targets, preds)
print(f"\nMAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")

# --- Plots ---
plt.figure(figsize=(10, 6))
plt.scatter(targets, preds, alpha=0.6)
plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
plt.xlabel("Actual Time Cycles")
plt.ylabel("Predicted Time Cycles")
plt.title("Improved MLP: Actual vs Predicted")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
plt.hist(preds, bins=30, color="blue", alpha=0.7)
plt.xlabel("Predicted Time Cycles")
plt.ylabel("Count")
plt.title("Distribution of Predictions")
plt.tight_layout()
plt.show()
