from data.mt5_data import fetch_mt5_data, preprocess
from data.dataset import TimeseriesDataset
from torch.utils.data import DataLoader

# Fetch data
df = fetch_mt5_data("XAUUSD", timeframe=1, n_bars=10000)
arr = preprocess(df)

# Dataset
dataset = TimeseriesDataset(arr, input_len=100, forecast_horizon=5)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for X, y in loader:
 print("Batch X:", X.shape)
 print("Batch y:", y.shape)
 break

from data.mt5_data import fetch_mt5_data, preprocess
from utils.save_utils import load_model, load_scaler, load_config

# Example usage
df = fetch_mt5_data("XAUUSD", timeframe=1, n_bars=2000)
arr = preprocess(df)

scaler = load_scaler("checkpoints/scaler.pkl")
arr_scaled = scaler.transform(arr)

config = load_config("checkpoints/config.json")
print("Loaded config:", config)







# requirements: torch, numpy, pandas, scikit-learn, pytorch-lightning(optional)
# Save as hybrid_pipeline.py and adapt data paths

import os
import math
import numpy as np
import pandas as pd
from glob import glob
from typing import Tuple, Dict, Any
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import random
import time

# ---------------------------
# Utilities
# ---------------------------
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seeds(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Save & load helpers
def save_checkpoint(state: Dict, path: str):
    torch.save(state, path)

def load_checkpoint(path: str, model: nn.Module, optimizer=None):
    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    if optimizer and "opt_state" in ckpt:
        optimizer.load_state_dict(ckpt["opt_state"])
    return ckpt

# ---------------------------
# Data: MultiTF Dataset (skeleton)
# ---------------------------
class MultiTFDataset(Dataset):
    """
    Each sample returns:
      - htf_seq: np.float32 [htf_len, hf_features]
      - ltf_seq: np.float32 [ltf_len, lf_features]  # 15M sequence for entry
      - mng_seq: np.float32 [mng_len, mg_features]  # 1M sequence for management (if trade open)
      - trade_state: np.float32 [state_dim]
      - labels: dict with 'htf_label', 'entry_label', 'mng_label'
    """
    def __init__(self, samples: pd.DataFrame, scalers: Dict[str, Any],
                 htf_len=600, ltf_len=80, mng_len=120, mode="entry"):
        # samples: DataFrame with pointers to where sequences live or precomputed arrays
        self.samples = samples.reset_index(drop=True)
        self.scalers = scalers
        self.htf_len = htf_len
        self.ltf_len = ltf_len
        self.mng_len = mng_len
        self.mode = mode

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples.loc[idx]
        # For example we stored file paths to precomputed numpy sequences:
        htf = np.load(row["htf_path"]).astype(np.float32)   # shape [htf_len, n_feats_htf]
        ltf = np.load(row["ltf_path"]).astype(np.float32)   # shape [ltf_len, n_feats_ltf]
        # mng might not exist for all rows; for training management, you'd have dataset filtered to trades
        mng = np.load(row["mng_path"]).astype(np.float32) if "mng_path" in row and row["mng_path"] else np.zeros((self.mng_len, 5), dtype=np.float32)
        # trade state
        trade_state = np.array(row.get("trade_state", np.zeros(6)), dtype=np.float32)
        labels = {
            "htf": int(row["htf_label"]),
            "entry": int(row["entry_label"]),
            "mng": int(row.get("mng_label", 0))
        }
        # Optionally apply scalers here (or you saved scaled arrays)
        return {
            "htf": torch.from_numpy(htf),
            "ltf": torch.from_numpy(ltf),
            "mng": torch.from_numpy(mng),
            "state": torch.from_numpy(trade_state),
            "labels": labels
        }
