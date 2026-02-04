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

# ----------------------------
# CONFIG
# ----------------------------
SEQ_LEN = 48
BATCH_SIZE = 64
EPOCHS = 20
LR = 0.001
QUANTILES = [0.1, 0.5, 0.9]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# 1. DATA GENERATION
# ----------------------------

def generate_data(n=5000):
    t = np.arange(n)
    s1 = 10 + np.sin(2*np.pi*t/24) + 0.3*np.random.randn(n)
    s2 = 5 + np.cos(2*np.pi*t/168) + 0.3*np.random.randn(n)
    s3 = 0.01*t + np.random.randn(n)
    return pd.DataFrame({"x1": s1, "x2": s2, "x3": s3})

df = generate_data()

# Differencing
df = df.diff().dropna()

scaler = MinMaxScaler()
scaled = scaler.fit_transform(df)

# ----------------------------
# 2. DATASET
# ----------------------------

class TimeDataset(Dataset):
    def _init_(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def _len_(self):
        return len(self.data)-self.seq_len

    def _getitem_(self, idx):
        x = self.data[idx:idx+self.seq_len]
        y = self.data[idx+self.seq_len,0]
        return torch.tensor(x,dtype=torch.float32), torch.tensor(y,dtype=torch.float32)

dataset = TimeDataset(scaled, SEQ_LEN)
train_size = int(0.8*len(dataset))
test_size = len(dataset)-train_size
train_ds, test_ds = torch.utils.data.random_split(dataset,[train_size,test_size])

train_loader = DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=True)
test_loader = DataLoader(test_ds,batch_size=BATCH_SIZE)

# ----------------------------
# 3. PINBALL LOSS
# ----------------------------

class PinballLoss(nn.Module):
    def _init_(self, quantiles):
        super()._init_()
        self.q = quantiles

    def forward(self, preds, target):
        losses=[]
        for i,q in enumerate(self.q):
            err = target - preds[:,i]
            loss = torch.max(q*err,(q-1)*err)
            losses.append(loss.unsqueeze(1))
        return torch.mean(torch.cat(losses,1))

# ----------------------------
# 4. MODEL
# ----------------------------

class LSTMQuantile(nn.Module):
    def _init_(self,input_size,hidden):
        super()._init_()
        self.lstm = nn.LSTM(input_size,hidden,batch_first=True)
        self.fc = nn.Linear(hidden,len(QUANTILES))

    def forward(self,x):
        o,_ = self.lstm(x)
        return self.fc(o[:,-1])

model = LSTMQuantile(3,64).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(),lr=LR)
criterion = PinballLoss(QUANTILES)

# ----------------------------
# 5. TRAINING
# ----------------------------

for e in range(EPOCHS):
    model.train()
    total=0
    for x,y in train_loader:
        x,y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds,y)
        loss.backward()
        optimizer.step()
        total+=loss.item()
    print(f"Epoch {e+1}/{EPOCHS}  Loss:{total/len(train_loader):.4f}")

# ----------------------------
# 6. EVALUATION
# ----------------------------

def evaluate(model,loader):
    model.eval()
    preds=[]
    trues=[]
    with torch.no_grad():
        for x,y in loader:
            p=model(x.to(DEVICE)).cpu().numpy()
            preds.append(p)
            trues.append(y.numpy())
    return np.vstack(preds),np.concatenate(trues)

P,Y = evaluate(model,test_loader)

median_pred=P[:,1]
rmse=np.sqrt(mean_squared_error(Y,median_pred))
mae=mean_absolute_error(Y,median_pred)

# Pinball Score
pinball=[]
for i,q in enumerate(QUANTILES):
    err=Y-P[:,i]
    pinball.append(np.mean(np.maximum(q*err,(q-1)*err)))

# Coverage
lower=P[:,0]
upper=P[:,2]
coverage=np.mean((Y>=lower)&(Y<=upper))

print("\n---- Neural Network Metrics ----")
print("RMSE:",rmse)
print("MAE:",mae)
for q,s in zip(QUANTILES,pinball):
    print(f"Pinball(q={q}):",s)
print("Coverage:",coverage)

# ----------------------------
# 7. BASELINE MODEL
# ----------------------------

series=df["x1"].values
train=series[:-test_size]
test=series[-test_size:]

hw=ExponentialSmoothing(train,trend="add",seasonal=None).fit()
baseline_pred=hw.forecast(test_size)

b_rmse=np.sqrt(mean_squared_error(test,baseline_pred))
b_mae=mean_absolute_error(test,baseline_pred)

print("\n---- Baseline (Exponential Smoothing) ----")
print("RMSE:",b_rmse)
print("MAE:",b_mae)

# ----------------------------
# 8. CALIBRATION CHECK
# ----------------------------

bins=np.linspace(0,1,10)
obs=[]
for b in bins:
    q_pred=np.quantile(P[:,1],b)
    obs.append(np.mean(Y<=q_pred))

print("\nCalibration (Expected vs Observed)")
for b,o in zip(bins,obs):
    print(round(b,2),"->",round(o,2))

print("\nFinished Successfully")
