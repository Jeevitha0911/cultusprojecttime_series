# ============================================================
# Advanced Time Series Forecasting with Uncertainty Quantification
# ============================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ============================================================
# CONFIGURATION
# ============================================================

SEQ_LEN = 48
BATCH_SIZE = 64
EPOCHS = 20
LR = 0.001
QUANTILES = [0.1, 0.5, 0.9]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# TASK 1: MULTIVARIATE DATA GENERATION
# ============================================================

def generate_data(n=6000):
    t = np.arange(n)
    s1 = 10 + np.sin(2*np.pi*t/24) + 0.3*np.random.randn(n)     # Daily seasonality
    s2 = 5 + np.cos(2*np.pi*t/168) + 0.3*np.random.randn(n)    # Weekly seasonality
    s3 = 0.01*t + np.random.randn(n)                           # Trend
    return pd.DataFrame({"x1": s1, "x2": s2, "x3": s3})

df = generate_data()

# Differencing
df = df.diff().dropna()

# Normalization
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df)

# ============================================================
# TASK 1: SEQUENCE DATASET
# ============================================================

class TimeDataset(Dataset):
    def _init_(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def _len_(self):
        return len(self.data) - self.seq_len

    def _getitem_(self, idx):
        X = self.data[idx:idx+self.seq_len]
        y = self.data[idx+self.seq_len, 0]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

dataset = TimeDataset(scaled, SEQ_LEN)

# Chronological Split
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_ds = torch.utils.data.Subset(dataset, range(train_size))
test_ds  = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE)

# ============================================================
# TASK 2: CUSTOM PINBALL LOSS
# ============================================================

class PinballLoss(nn.Module):
    def _init_(self, quantiles):
        super()._init_()
        self.quantiles = quantiles

    def forward(self, preds, target):
        losses = []
        for i, q in enumerate(self.quantiles):
            err = target - preds[:, i]
            losses.append(torch.max(q*err, (q-1)*err).unsqueeze(1))
        return torch.mean(torch.cat(losses, dim=1))

# ============================================================
# TASK 2: LSTM QUANTILE MODEL
# ============================================================

class LSTMQuantile(nn.Module):
    def _init_(self, input_size, hidden):
        super()._init_()
        self.lstm = nn.LSTM(input_size, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, len(QUANTILES))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])

model = LSTMQuantile(3, 64).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = PinballLoss(QUANTILES)

# ============================================================
# TASK 2: TRAINING
# ============================================================

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}  Loss: {total_loss/len(train_loader):.4f}")

# ============================================================
# TASK 3: ROLLING-ORIGIN EVALUATION
# ============================================================

def evaluate(model, loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            p = model(x.to(DEVICE)).cpu().numpy()
            preds.append(p)
            trues.append(y.numpy())
    return np.vstack(preds), np.concatenate(trues)

P, Y = evaluate(model, test_loader)

# ============================================================
# TASK 3: POINT FORECAST METRICS
# ============================================================

median_pred = P[:,1]
rmse = np.sqrt(mean_squared_error(Y, median_pred))
mae  = mean_absolute_error(Y, median_pred)

# ============================================================
# TASK 3: PROBABILISTIC METRICS
# ============================================================

pinball_scores = []
for i, q in enumerate(QUANTILES):
    err = Y - P[:, i]
    score = np.mean(np.maximum(q*err, (q-1)*err))
    pinball_scores.append(score)

lower = P[:,0]
upper = P[:,2]
coverage = np.mean((Y >= lower) & (Y <= upper))

print("\n---- Neural Network Metrics ----")
print("RMSE:", rmse)
print("MAE :", mae)
for q, s in zip(QUANTILES, pinball_scores):
    print(f"Pinball(q={q}): {s}")
print("Overall Coverage:", coverage)

# ============================================================
# TASK 3: BASELINE (EXPONENTIAL SMOOTHING)
# ============================================================

series = df["x1"].values
train_series = series[:train_size]
test_series  = series[train_size:]

hw = ExponentialSmoothing(train_series, trend="add", seasonal=None).fit()
baseline_pred = hw.forecast(len(test_series))

b_rmse = np.sqrt(mean_squared_error(test_series, baseline_pred))
b_mae  = mean_absolute_error(test_series, baseline_pred)

print("\n---- Baseline Metrics ----")
print("RMSE:", b_rmse)
print("MAE :", b_mae)

# ============================================================
# TASK 4: CALIBRATION ANALYSIS
# ============================================================

bins = np.linspace(0, 1, 10)
observed = []

for b in bins:
    q_val = np.quantile(P[:,1], b)
    observed.append(np.mean(Y <= q_val))

print("\nCalibration (Expected vs Observed)")
for e, o in zip(bins, observed):
    print(round(e,2), "->", round(o,2))

print("\nPROJECT EXECUTION COMPLETED SUCCESSFULLY")
