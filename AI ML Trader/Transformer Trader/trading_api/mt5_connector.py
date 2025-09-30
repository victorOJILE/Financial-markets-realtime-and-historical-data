import MetaTrader5 as mt5
import pandas as pd
import json
import os 
from datetime import datetime
from scripts.data_preparation import create_features

# === Trading params (tweak later) ===
CONF_THRESHOLD = 0.6 # min confidence to act 
RISK_PER_TRADE = 0.01 # 1% of balance 
MAX_LOTS = 5.0 # safety cap 
MAGIC = 123456 
DEVIATION = 20

# === Equity protection parameters ===
DAILY_LOSS_LIMIT = 0.02 # stop trading for the day if loss >= 2% of starting balance
MAX_DRAWDOWN_LIMIT = 0.05 # stop trading if drawdown from peak equity >= 5% 
RISK_STATE_FILE = 'risk_state.json'

class MT5Connector:
 def __init__(self, login, password, server, symbol, scaler=False):
  self.login = login
  self.password = password
  self.server = server
  self.symbol = symbol
  self.initialized = False
  self.connect()
  self.mt5 = mt5
  
  if scaler: 
   with open("data/scaler4h.pkl", "rb") as f:
    self.scaler4h = pickle.load(f)
   
   with open("data/scaler15m.pkl", "rb") as f:
    self.scaler15m = pickle.load(f)
   
   if (not self.scaler15m) or) not self.scaler4h): 
    raise SystemExit("⚠️ One or both timeframe scalers can't be loaded.")
 
 def connect(self):
  self.initialized = mt5.initialize(login=self.login, password=self.password, server=self.server)
  if not self.initialized: 
   raise SystemExit("MetaTrader5 initialization failed, error code =", mt5.last_error())
  
  # === Ensure symbol is available ===
  if not mt5.symbol_select(self.symbol, True):
   raise SystemExit(f"Failed to select symbol {self.symbol}")
  
  print("Metatrader 5 connections successful")
  
  return mt5
 
 def get_live_data(self, timeframe, num_bars):
  if not self.initialized:
   self.connect()
   
   rates = mt5.copy_rates_from_pos(self.symbol, timeframe, 0, num_bars)
   if rates is None:
    print(f"Failed to get data for {self.symbol} on {timeframe}, error code =", mt5.last_error())
    return None
   scaler = self.scaler15m if timeframe == mt5.TIMEFRAME_M15 else self.scaler4h
   
   df, _ = create_features(pd.DataFrame(rates), scaler)
   df['time'] = pd.to_datetime(df['time'], unit='s')
   
   return df
 
 def send_order(self, order_type, tick):
  price = tick.ask if order_type == 1 else tick.bid
  volume = ""
  
  
  # Create a trading request
  request = {
   "action": mt5.TRADE_ACTION_DEAL,
   "symbol": self.symbol,
   "volume": volume,
   "deviation": DEVIATION,
   "type": mt5.ORDER_TYPE_BUY if order_type == 1 else mt5.ORDER_TYPE_SELL,
   "price": price,
   "magic": MAGIC,
  }
  
  result = mt5.order_send(request)
  if result.retcode != mt5.TRADE_RETCODE_DONE:
   print(f"{'BUY' if order_type == 1 else 'SELL'} order failed, error code: {result.retcode}")
  
  print(f"Sent {'BUY' if order_type == 1 else 'SELL'} order based on Entry Model signal.")

 def get_open_positions(self):
  return mt5.positions_get(symbol=self.symbol)
 
 def close_position(ticket, volume): 
  # Find position by ticket and send opposite order to close volume 
  pos = None 
  positions = mt5.positions_get() 
  for p in positions:
   if p.ticket == ticket: 
    pos = p 
    break
  else: 
   print(f"Position {ticket} not found")
   return
  
  symbol = pos.symbol
  price = mt5.symbol_info_tick(symbol).bid if pos.type == 0 else mt5.symbol_info_tick(symbol).ask
  order_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
    
  request = {
    "action": mt5.TRADE_ACTION_DEAL,
    "symbol": symbol,
    "volume": float(volume) if volume != None else pos.volume,
    "type": order_type,
    "price": float(price),
    "comment": "Transformer_close",
    "type_filling": mt5.ORDER_FILLING_IOC,
  }
  result = mt5.order_send(request)
  if result.retcode != mt5.TRADE_RETCODE_DONE:
   print(f"Close order failed, error code: {result.retcode}")
   return False
  return True

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
def get_today_period(): 
 # using UTC day boundary for simplicity
 today = datetime.utcnow().date() 
 start = datetime(today.year, today.month, today.day) 
 end = datetime.utcnow() 
 return start, end
# TODO:
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
