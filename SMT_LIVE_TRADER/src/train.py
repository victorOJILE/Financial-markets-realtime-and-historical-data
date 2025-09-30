import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from model import SMTTransformer, LABELS
from datetime import datetime

MAX_CANDLES = 50
TIMEFRAMES = ["M5", "M15", "M30"]
void label_from_smt_logic(string &pairs[]) {
 string symbol1 = pairs[0];
 string symbol2 = pairs[1];
 
 double highs1[];
 double highs2[];
 double lows1[];
 double lows2[];
 
 if (CopyHigh(symbol1, tf, 0, bars, highs1) < bars || CopyHigh(symbol2, tf, 0, bars, highs2) || CopyLow(symbol1, tf, 0, bars, lows1) < bars || CopyLow(symbol2, tf, 0, bars, lows2) < bars) {
  Print("Error copying rates: ", GetLastError());
  return;
 }
 
 double lastHigh1 = iHigh(symbol1, tf, 1);
 double lastHigh2 = iHigh(symbol2, tf, 1);
 double curHigh = 0.0;
 double curHigh2 = 0.0;
 for(int i = 4; i < bars -2; i++) {
  if(!IsFractal(highs1, i, true) && !IsFractal(highs2, i, true)) continue;
  if(curHigh <= highs1[i] && curHigh2 <= highs2[i]) continue;
  curHigh = MathMax(curHigh, highs1[i]);
  curHigh2 = MathMax(curHigh2, highs2[i]);
  
  if((lastHigh1 > curHigh && lastHigh1 < curHigh) || (lastHigh2 < curHigh2 and lastHigh2 > curHigh2)) EnterTrade(symbol1, symbol2, false);
 }
 
 double lastLow1 = iLow(symbol1, tf, 1);
 double lastLow2 = iLow(symbol2, tf, 1);
 double curLow = 0.0;
 double curLow2 = 0.0;
 for(int i = 4; i < bars -2; i++) {
  if(!IsFractal(lows1, i, true) && !IsFractal(lows2, i, true)) continue;
  if(curLow <= lows1[i] && curLow2 <= lows2[i]) continue;
  curLow = MathMin(curLow, lows1[i]);
  curLow2 = MathMin(curLow2, lows2[i]);
  
  if((lastLow1 > curLow && lastLow1 < curLow) || (lastLow2 < curLow2 && lastLow2 > curLow2)) EnterTrade(symbol1, symbol2, true);
 }
}

# Dummy label logic for now
def label_from_smt_logic(eur_candles, gbp_candles):
    # TODO: Replace with SMT divergence logic
    if eur_candles[-1, 1] > eur_candles[-2, 1] and gbp_candles[-1, 1] < gbp_candles[-2, 1]:
        return 0  # Buy
    elif eur_candles[-1, 2] < eur_candles[-2, 2] and gbp_candles[-1, 2] > gbp_candles[-2, 2]:
        return 1  # Sell
    return 2  # Hold

def load_ohlc(pair, tf):
    path = f"../data/{pair}_{tf}.csv"
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['timestamp_delta'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()
    return df[['open', 'high', 'low', 'close', 'timestamp_delta']].values

class ForexDataset(Dataset):
    def __init__(self):
        self.inputs = []
        self.labels = []

        eur = {tf: load_ohlc("EURUSD", tf) for tf in TIMEFRAMES}
        gbp = {tf: load_ohlc("GBPUSD", tf) for tf in TIMEFRAMES}

        min_len = min(len(eur["M5"]), len(gbp["M5"]))
        for i in range(MAX_CANDLES, min_len):
            tensor = []
            for tf in TIMEFRAMES:
                eur_slice = eur[tf][i - MAX_CANDLES:i]
                gbp_slice = gbp[tf][i - MAX_CANDLES:i]
                tensor.append((eur_slice, gbp_slice))
            eur_stack = np.stack([x[0] for x in tensor])
            gbp_stack = np.stack([x[1] for x in tensor])
            full_tensor = np.stack([eur_stack, gbp_stack])  # [pair, tf, candle, features]
            label = label_from_smt_logic(eur_stack[-1], gbp_stack[-1])
            self.inputs.append(torch.tensor(full_tensor, dtype=torch.float32))
            self.labels.append(label)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

def train_model():
    dataset = ForexDataset()
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = SMTTransformer()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(10):
        total_loss = 0
        for x, y in loader:
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "../models/smt_model.pt")
    print("✅ Model trained and saved to models/smt_model.pt")

if __name__ == "__main__":
    train_model()