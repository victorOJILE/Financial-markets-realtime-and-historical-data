import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import json
import os 
import time
from datetime import datetime
import pickle

from tensorflow.keras.models import Sequential, load_model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# === CONFIG ===
SYMBOL = "XAUUSD"
TIMEFRAME = mt5.TIMEFRAME_M30
WINDOW = 200 # timesteps (200)
FEATURES = ["open", "high", "low", "close", "obv", "fi", "slope"]
CLASSES = ["Buy", "Sell", "Hold"]
MODEL_PATH = "stacked_lstm_200.h5"

# === Trading params (tweak later) ===
CONF_THRESHOLD = 0.6 # min confidence to act 
RISK_PER_TRADE = 0.01 # 1% of balance 
MAX_LOTS = 5.0 # safety cap 
MAGIC = 123456 
DEVIATION = 20

# === Equity protection parameters ===
DAILY_LOSS_LIMIT = 0.02 # stop trading for the day if loss >= 2% of starting balance
MAX_DRAWDOWN_LIMIT = 0.10 # stop trading if drawdown from peak equity >= 10% 
RISK_STATE_FILE = 'risk_state.json'

# === MT5 INIT ===
if not mt5.initialize(): 
  raise SystemExit("MT5 initialize() failed")

# === Ensure symbol is available ===
if not mt5.symbol_select(SYMBOL, True): 
  raise SystemExit(f"Failed to select symbol {SYMBOL}")

# === PERSISTENT RISK STATE ===
def load_risk_state(): 
  if os.path.exists(RISK_STATE_FILE): 
    with open(RISK_STATE_FILE, 'r') as f:
      state = json.load(f) # convert datetimes if present 
      return state
  # default state 
  info = mt5.account_info() 
  start_balance = info.balance if info else 0.0 
  
  return {
    'trading_enabled': True,
    'start_of_day': datetime.utcnow().strftime('%Y-%m-%d'),
    'starting_balance': start_balance,
    'peak_equity': info.equity if info else start_balance,
    'last_saved': datetime.utcnow().isoformat()
  }

def save_risk_state(state):
  state['last_saved'] = datetime.utcnow().isoformat() 
  with open(RISK_STATE_FILE, 'w') as f: json.dump(state, f)

# === UTILITIES === 
def get_history(): 
  rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, WINDOW)
  df = pd.DataFrame(rates)
  df['time'] = pd.to_datetime(df['time'], unit='s') 
  return df

def get_today_period(): 
  # using UTC day boundary for simplicity
  today = datetime.utcnow().date() 
  start = datetime(today.year, today.month, today.day) 
  end = datetime.utcnow() 
  return start, end

def get_realized_pnl_today(): 
  start, end = get_today_period()
  # fetch deals (might be large; narrow by time)
  deals = mt5.history_deals_get(start, end) 
  if deals is None: 
    return 0.0 
  pnl = 0.0 
  for d in deals: 
    try: 
      # include only deals matching our magic/order comment if available 
      # Many brokers don't store magic in deals; filter by comment or symbol if needed. 
      if hasattr(d, 'magic') and d.magic == MAGIC: 
        pnl += d.profit 
    except Exception: 
      continue 
  return pnl

def update_peak_equity(state):
  info = mt5.account_info()
  if info is None:
    return state 
  equity = info.equity
  if equity > state.get('peak_equity', 0): 
    state['peak_equity'] = equity 
  return state

def current_drawdown(state):
  info = mt5.account_info()
  if info is None: 
    return 0.0 
  peak = state.get('peak_equity', info.equity) 
  return (peak - info.equity) / peak if peak > 0 else 0.0

# === BROKER-ACCURATE LOT SIZE ===
def calculate_lot_size(stop_loss_pips, risk_percent, risk_amount): 
  sym = mt5.symbol_info(SYMBOL)
  if sym is None:
    raise RuntimeError('Symbol info not available')
  
  # stop_loss_pips is expressed in pips (broker-dependent). For XAUUSD we assume 0.01 pip = 1 point.
  sl_price_distance = stop_loss_pips * sym.point
  
  # Determine currency of account vs symbol quote currency. We'll compute value per point per lot using contract_size & tick_value if available
  # Many brokers provide trade_contract_size and tick_value
  if hasattr(sym, "trade_tick_value") and hasattr(sym, "trade_tick_size"):
    # value per 1 point (price unit) for 1 lot
    value_per_point_per_lot = sym.trade_tick_value / sym.trade_tick_size
  elif hasattr(sym, "trade_contract_size"):
    # approximate: point value = trade_contract_size * point
    # This is crude and depends on instrument. Users should verify with broker calculation.
    value_per_point_per_lot = sym.trade_contract_size * sym.point
  else:
    # fallback: assume $1 per pip per lot (risky), user must verify with broker
    value_per_point_per_lot = 1.0
  
  risk_per_lot = sl_price_distance * value_per_point_per_lot
  if risk_per_lot <= 0:
    return 0.0
    
  raw_lots = risk_amount / risk_per_lot
    
  # enforce broker limits
  vol_step = getattr(sym, 'volume_step', 0.01)
  
  lots = max(sym.volume_min, min(raw_lots, sym.volume_max))
  # round to step
  if vol_step > 0:
    lots = np.floor(lots / vol_step) * vol_step
  lots = round(lots, 2)
  return lots

# === ORDER & TRADE HELPERS === 
def place_order(tick, direction, scale, sl_percent, tp_percent):
  # choose entry price based on direction
  entry_price = tick.ask if direction == 0 else tick.bid
  
  # compute SL/TP as percent distances from price (model predicted normalized values); this mapping must match training
  sl_price = entry_price * (1 - sl_percent) if direction == 0 else entry_price * (1 + sl_percent)
  tp_price = entry_price * (1 + tp_percent) if direction == 0 else entry_price * (1 - tp_percent)
  
  acc = mt5.account_info()
  if acc is None:
    raise RuntimeError('Unable to get account info')
  
  risk_amount = acc.balance * RISK_PER_TRADE * scale
  
  # compute lots from sl distance and risk amount (placeholder implementation)
  # TODO: with lots
  
  lots = calculate_lot_size(sl_price, entry_price, risk_amount)
  if lots <= 0:
    print('Computed lot size is zero, skipping trade')
    return
  
  request = {
    "action": mt5.TRADE_ACTION_DEAL,
    "symbol": SYMBOL,
    "volume": float(lots),
    "type": mt5.ORDER_TYPE_BUY if direction == 0 else mt5.ORDER_TYPE_SELL,
    "price": float(entry_price),
    "sl": float(sl_price),
    "tp": float(tp_price),
    "deviation": DEVIATION,
    "magic": MAGIC,
    "comment": "LSTM_auto",
    "type_filling": mt5.ORDER_FILLING_IOC,
  }
  
  res = mt5.order_send(request)
  if res is None:
    print("order_send returned None")
    return None
  
  print(f"Order send result: {res}")
  return res

def manage_open_trades(direction, conf): 
  # example: close trades when model strongly signals opposite or confidence collapses 
  positions = mt5.positions_get(symbol=SYMBOL)
  if positions is None or len(positions) == 0:
    return
  
  for pos in positions:
    ticket = pos.ticket
    pos_type = pos.type  # 0=buy,1=sell
    # if model signals opposite with high confidence -> close position
    if (pos_type == 0 and direction == 1 and conf > 0.7) or (pos_type == 1 and direction == 0 and conf > 0.7):
      close_position(ticket, pos.volume)
      print(f"Closed opposite trade ticket {ticket} due to model signal")
    # optional: scale down or partial close if confidence collapsed
    if conf < 0.3:
      # partial close 50%
      new_vol = round(pos.volume * 0.5, 2)
      close_position(ticket, pos.volume - new_vol)
      print(f"Partial close for ticket {ticket} due to low confidence")

def close_position(ticket, volume): 
  # Find position by ticket and send opposite order to close volume 
  pos = None 
  positions = mt5.positions_get() 
  for p in positions:
    if p.ticket == ticket: 
      pos = p 
      break 
    if pos is None: 
      print(f"Position {ticket} not found") 
      return

  symbol = pos.symbol
  price = mt5.symbol_info_tick(symbol).bid if pos.type == 0 else mt5.symbol_info_tick(symbol).ask
  order_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
    
  request = {
    "action": mt5.TRADE_ACTION_DEAL,
    "symbol": symbol,
    "volume": float(volume),
    "type": order_type,
    "price": float(price),
    "deviation": DEVIATION,
    "magic": MAGIC,
    "comment": "LSTM_close",
    "type_filling": mt5.ORDER_FILLING_IOC,
  }
  res = mt5.order_send(request)
  print(f"Close order result: {res}")
  return res

# === RISK CHECK BEFORE TRADING ===
def check_risk_before_trade():
  # reload state from disk in case external change 
  risk_state = load_risk_state()

  if not risk_state.get('trading_enabled', True):
    print('Trading disabled by risk manager')
    return False
  
  # update peak equity
  risk_state = update_peak_equity(risk_state)
  
  # check drawdown
  dd = current_drawdown(risk_state)
  if dd >= MAX_DRAWDOWN_LIMIT:
    print(f'Pausing trading: drawdown {dd:.2%} >= limit {MAX_DRAWDOWN_LIMIT:.2%}')
    risk_state['trading_enabled'] = False
    save_risk_state(risk_state)
    return False
  
  # check daily loss
  start, end = get_today_period()
  # realized pnl for today (matching our magic) - may require broker support
  pnl_today = get_realized_pnl_today(MAGIC)
  if pnl_today <= -abs(DAILY_LOSS_LIMIT * risk_state.get('starting_balance', 0)):
    print(f'Pausing trading: realized PnL today {pnl_today:.2f} <= daily loss limit')
    risk_state['trading_enabled'] = False
    save_risk_state(risk_state)
    return False
  
  return True

# === CREATE SUPERVISED DATA (Sliding Windows) ===
def create_sequences(features, labels, window=200): 
  X, y = [], [] 
  for i in range(len(features) - window):
    X.append(features[i:i+window]) 
    y.append(labels[i+window]) 
  return np.array(X), np.array(y)

def calc_obv(df):
  obv = [0]
  for i in range(1, len(df)):
    if df['close'].iloc[i] > df['close'].iloc[i-1]:
      obv.append(obv[-1] + df['tick_volume'].iloc[i])
    elif df['close'].iloc[i] < df['close'].iloc[i-1]:
      obv.append(obv[-1] - df['tick_volume'].iloc[i])
    else:
      obv.append(obv[-1])
  df['obv'] = obv
  return df

def calc_fi(df):
  df['fi'] = (df['close'] - df['close'].shift(1)) * df['tick_volume']
  df['fi'].fillna(0, inplace=True)
  return df

def slope_numpy(y):
  x = np.arange(len(y))
  slope, _ = np.polyfit(x, y, 1)
  return slope

def calc_slope(df, window=3):
  slopes = []
  for i in range(len(df)):
    if i < window - 1:
      slopes.append(0)
    else:
      y = df['close'].iloc[i-window+1:i+1].values
      slopes.append(slope_numpy(y))
  df['slope'] = slopes
  return df

# === TRAIN MODEL (optional) === 
def train_and_save_model():
  print("Fetching history for training...") 
  data = get_history()

  closes = data['close'].values
  labels = []
  for i in range(len(closes)-1):
    if closes[i+1] > closes[i]:
      labels.append(0)  # Buy
    elif closes[i+1] < closes[i]:
      labels.append(1)  # Sell
    else:
      labels.append(2)  # Hold
  labels.append(2)

  # === FEATURE ENGINEERING ===
  data = calc_obv(data)
  data = calc_fi(data)
  data = calc_slope(data, window=3)
  
  # === FEATURE SCALING ===
  scaler = MinMaxScaler(feature_range=(0,1)) 
  scaled = scaler.fit_transform(data[FEATURES].values)
  
  X, y = create_sequences(scaled, labels, WINDOW)
  y = to_categorical(y[:len(X)], num_classes=len(CLASSES))
  
  print("X shape:", X.shape, " y shape:", y.shape)
  
  # === TRAIN/TEST SPLIT ===
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
  
  # === BUILD STACKED LSTM MODEL ===
  model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(WINDOW, len(FEATURES))),
    Dropout(0.2),
    LSTM(64),
    Dense(32, activation='relu'),
    Dense(len(CLASSES), activation='softmax')
  ])
  
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
  model.summary()
  
  # === CALLBACKS ===
  early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,          # stop if no improvement for 3 epochs
    restore_best_weights=True
  )
  checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor='val_loss',
    save_best_only=True,
    verbose=1
  )
  # === TRAIN ===
  model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32, callbacks=[early_stop, checkpoint], verbose=1)
  
  model.save(MODEL_PATH)
  print(f"Model saved to {MODEL_PATH}")
  with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
    print(f"Scaler saved to {f}")

# === LIVE LOOP: on candle close === 
def run_live(symbol): 
  model = None
  scaler = None
  if os.path.exists(MODEL_PATH):
    try:
      model = load_model(MODEL_PATH)
    except Exception as e:
      raise SystemExit("⚠️ Failed to load model: {e}")
  else:
    raise SystemExit("⚠️ No trained model found, skipping trading loop.")
  
  with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
  if scaler is None:
    raise RuntimeError("Scaler not loaded. Train model first and save scaler.pkl")
  
  last_time = None
  last_position = None
  # ensure risk_state has sensible starting values 
  risk_state = load_risk_state()
  save_risk_state(risk_state)
  
  # You can replace these with a management head outputs when training multi-head model
  sl_percent = 0.002  # 0.2%
  tp_percent = 0.004  # 0.4%
  
  print('Starting live loop')
  while True:
    try:
      tick = mt5.symbol_info_tick(SYMBOL)
      if tick is None:
        print("No market tick info; skipping")
        sleep(10) 
        continue
      
      # preprocess latest window
      df = get_history()
      latest_candle_time = df['time'].iloc[-1]

      # only run on new candle
      if last_time is None or latest_candle_time != last_time:
        last_time = latest_candle_time
        print(f"New candle closed at {last_time}")
      
        # update risk state timing: reset starting balance at UTC day boundary 
        today_str = datetime.utcnow().strftime('%Y-%m-%d')
        if risk_state.get('start_of_day') != today_str: 
          # new day: reset starting balance and enable trading 
          info = mt5.account_info()
          risk_state['start_of_day'] = today_str
          risk_state['starting_balance'] = info.balance if info else risk_state.get('starting_balance', 0) 
          risk_state['trading_enabled'] = True 
          risk_state['peak_equity'] = info.equity if info else risk_state.get('peak_equity', 0)
          save_risk_state(risk_state)
          
        # risk gating
        if not check_risk_before_trade():
          time.sleep(10)
          continue
          
        # predict
        features = df[FEATURES].values[-WINDOW:]  # last 200 candles
        scaled_features = scaler.transform(features)

        X = scaled_features.reshape(1, WINDOW, -1)  # shape for LSTM
        
        preds = model.predict(X, verbose=0)
        confidence = float(np.max(preds[0]))
        direction = int(np.argmax(preds[0]))
        
        print(f"{datetime.utcnow().isoformat()} - Signal: {CLASSES[direction]} conf={confidence:.3f}")
  
        if confidence < CONF_THRESHOLD:
          print('Low confidence threshold!')
          time.sleep(10)
          continue
        
        scale = 1.0 if confidence > 0.7 else 0.5 if confidence > 0.6 else 0.0
        if(last_position == direction): 
          # TODO: Last position can be current position in times of continuation trend
          print("Skipping this entry! Previous entry was same signal.")
        
        if scale > 0 and last_position != direction: 
          place_order(tick, direction, scale, sl_percent, tp_percent)
          manage_open_trades(direction, confidence)
          last_position = direction
      
      # small delay before next loop
      time.sleep(10)

    except KeyboardInterrupt:
      print('Stopping live loop')
      break
    except Exception as e:
      print('Error in live loop:', e)
      time.sleep(10)

# === ENTRY === 
if __name__ == '__main__':
  global risk_state
  if os.path.exists(MODEL_PATH):
    # run live trading
    run_live(SYMBOL)
  else: 
    # Optionally train model (uncomment if you want to train here): 
    # train_and_save_model()
    # run_live(SYMBOL)
