import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime

tf = {
 "M5": mt5.TIMEFRAME_M5,
 "M15": mt5.TIMEFRAME_M15,
 "H4": mt5.TIMEFRAME_H4
}

def fetch_mt5_data(timeframe=mt5.M5, n_bars=10000):
 if not mt5.initialize():
  raise RuntimeError("MT5 initialization failed")

 rates = mt5.copy_rates_from_pos("EURUSD", timeframe, 0, n_bars)
 mt5.shutdown()

 df = pd.DataFrame(rates)
 df['time'] = pd.to_datetime(df['time'], unit='s')
 return df

def preprocess(df, features=['open','high','low','close','tick_volume']):
 arr = df[features].values.astype(np.float32)
 return arr
