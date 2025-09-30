import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import torch

MAX_CANDLES = 50
TIMEFRAMES = {
 "M5": mt5.TIMEFRAME_M5,
 "M15": mt5.TIMEFRAME_M15,
 "M30": mt5.TIMEFRAME_M30
}
LABELS = ["Buy", "Sell", "Hold"]

def get_ohlc(symbol, timeframe, count=MAX_CANDLES):
 rates = mt5.copy_rates_from_pos(symbol, TIMEFRAMES[timeframe], 0, count)
 df = pd.DataFrame(rates)
 df['time'] = pd.to_datetime(df['time'], unit='s')
 df['timestamp_delta'] = (df['time'] - df['time'].iloc[0]).dt.total_seconds()
 return df[['open', 'high', 'low', 'close', 'timestamp_delta']].values

def prepare_live_tensor(pair1, pair2, timeframes):
 data = []
 for pair in [pair1, pair2]:
  tf_data = []
  for tf in timeframes:
   tf_data.append(get_ohlc(pair, tf))
   data.append(np.stack(tf_data, axis=0))
 return torch.tensor(np.stack(data, axis=0), dtype=torch.float32)

def execute_trade(symbol, signal, volume=0.1):
    order_type = {
        0: mt5.ORDER_TYPE_BUY,
        1: mt5.ORDER_TYPE_SELL
    }
    if signal not in [0, 1]: return # hold

    price = mt5.symbol_info_tick(symbol).ask if signal == 0 else mt5.symbol_info_tick(symbol).bid
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type[signal],
        "price": price,
        "deviation": 10,
        "magic": 234000,
        "comment": "SMT_AI_TRADE",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    print(f"Trade executed: {result}")

