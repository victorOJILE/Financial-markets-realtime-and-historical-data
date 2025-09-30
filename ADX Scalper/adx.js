//+------------------------------------------------------------------+
//|                                                   ADX_Scalper.mq5 |
//|                      Copyright 2025, YOUR_NAME |
//|                                       https://www.yourwebsite.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, OJILE VICTOR"
#property link "https://www.yourwebsite.com"
#property version "1.00"
#property description "Advanced ADX Scalper EA"
#property strict

input group "Main Parameters"
input ENUM_TIMEFRAME tf = PERIOD_M2; // Trading timeframe
input double LotSize = 0.01; // Initial lot size
input int TP_Pips = 15; // Take Profit in pips
input int SL_Pips = 50; // Stop Loss in pips
input int MagicNumber = 12345; // Magic number for trades
input double ADX_Threshold = 26.0; // ADX value to enable trading
input int ADX_Period = 14; // ADX period
input double LotMultiplier = 1.5; // Lot multiplier

input group "Loss Recovery"
input bool EnableRecovery = true; // Enable lot multiplier
input int MaxRecoveryTrades = 5; // Max number of recovery trades
input int RecoveryTP_Pips = 10; // Take Profit for recovery trades

input group "Volatile Moves"
input double ADX_LIT = 30.0; // ADX_Lot_Increase_Threshold, +DI/-DI value to increase lot size
input double ADX_LITMax = 40.0; // ADX_Lot_Increase_Threshold_Max, +DI/-DI max value to increase lot size
input int CandleCloseSL = 5; // Small pip value for trailing stop
input int LongCandlePips = 20; // Pips for "long candle" definition
input int VeryLongCandlePips = 40; // Pips for "very long candle" definition
input int TSL_MinPips = 2; // Minimum pips to update the trailing stop

enum TradeType = { NONE, BUY, SELL }

//--- Global Variables ---
int adxHandle;
double adxValues[2];
double plusDIValues[2];
double minusDIValues[2];
TradeType lastTradeType = NONE;

// Track if we're in a recovery state
bool inRecoveryMode = false;

int OnInit() {
 adxHandle = iADX(_Symbol, tf, ADX_Period);
 
 if (adxHandle == INVALID_HANDLE) {
  Print("Failed to create ADX indicator handles");
  return (INIT_FAILED);
 }
 
 return (INIT_SUCCEEDED);
}

void OnDeinit(const int reason) {
 IndicatorRelease(adxHandle);
}

void OnTick() {
 if (TerminalInfoInteger(TERMINAL_STATE) == TERMINAL_STATE_BUSY || !IsNewBar())
  return;
 
 ManageTrailingStop();
 
 //--- Get ADX values ---
 if (CopyBuffer(adxHandle, 0, 0, 2, adxValues) <= 0 ||
  CopyBuffer(adxHandle, 1, 0, 2, plusDIValues) <= 0 ||
  CopyBuffer(adxHandle, 2, 0, 2, minusDIValues) <= 0) {
  Print("Failed to get ADX buffer data");
  return;
 }
 
 double plusDI = plusDIValues[1];
 double minusDI = minusDIValues[1];
 
 double currentLot = GetTradeLotSize(plusDI, minusDI);
 
 //--- Entry/Recovery Logic ---
 // Trade enabled
 if (adxValues[1] >= ADX_Threshold) {
  // Reset recovery mode when a strong ADX signal returns
  inRecoveryMode = false;
  
  // --- Buy Signal ---
  if (plusDI > minusDI && plusDI >= ADX_Threshold) {
   // If a SELL signal was the last one, it's a signal switch.
   if (lastTradeType == SELL && OpenPositionsCount(MagicNumber, ORDER_TYPE_SELL) > 0) ManageRecoveryTrades(ORDER_TYPE_SELL);
   
   OpenPosition(ORDER_TYPE_BUY, LotSize, false);
   lastTradeType = BUY;
  }
  // --- Sell Signal ---
  else if (minusDI > plusDI && minusDI > ADX_Threshold) {
   // If a BUY signal was the last one, it's a signal switch.
   if (lastTradeType == BUY && OpenPositionsCount(MagicNumber, ORDER_TYPE_BUY) > 0) ManageRecoveryTrades(ORDER_TYPE_BUY);
   
   OpenPosition(ORDER_TYPE_SELL, LotSize, false);
   lastTradeType = SELL;
  }
 } else {
  // ADX is below threshold, Trade disabled
  if (OpenPositionsCount(MagicNumber, ORDER_TYPE_BUY) == 0 && OpenPositionsCount(MagicNumber, ORDER_TYPE_SELL) == 0) {
   // No open trades, so no need to manage recovery
   inRecoveryMode = false;
  } else {
   // We have open trades, check if they are losing trades that need a recovery attempt
   if (lastTradeType == BUY && OpenPositionsCount(MagicNumber, ORDER_TYPE_BUY) > 0) {
    ManageRecoveryTrades(ORDER_TYPE_BUY);
    inRecoveryMode = true;
   } else if (lastTradeType == SELL && OpenPositionsCount(MagicNumber, ORDER_TYPE_SELL) > 0) {
    ManageRecoveryTrades(ORDER_TYPE_SELL);
    inRecoveryMode = true;
   } else {
    // If there are trades but they are not losing trades from a specific type (e.g. they are break-even or have no TP), just manage them
    // (no change needed here, as the TP is set at the open, and this is for recovery)
   }
  }
 }
}

void ManageRecoveryTrades(ENUM_ORDER_TYPE losing_trade_type) {
 int openTradesCount = OpenPositionsCount(MagicNumber, losing_trade_type);
 
 if (openTradesCount > 0 && openTradesCount <= MaxRecoveryTrades) {
  double recoveryLot = LotSize * pow(LotMultiplier, openTradesCount);
  
  UpdateAllOpenPositionsTP(losing_trade_type);
  
  // Todo: Use TP from recovery position as update for previous losing positions
  OpenPosition(losing_trade_type, recoveryLot, true);
 }
}

void UpdateAllOpenPositionsTP(ENUM_ORDER_TYPE losing_trade_type) {
 double point = SymbolInfoDouble(Symbol(), SYMBOL_POINT);
 double new_tp_price = 0;
 
 if (losing_trade_type == ORDER_TYPE_SELL) {
  new_tp_price = SymbolInfoDouble(Symbol(), SYMBOL_BID) - RecoveryTP_Pips * point;
 } else if (losing_trade_type == ORDER_TYPE_BUY) {
  new_tp_price = SymbolInfoDouble(Symbol(), SYMBOL_ASK) + RecoveryTP_Pips * point;
 }
 
 for (int i = PositionsTotal() - 1; i >= 0; i--) {
  ulong ticket = PositionGetTicket(i);
  
  if (PositionSelectByTicket(ticket)) {
   if (PositionGetInteger(POSITION_TYPE) == losing_trade_type && PositionGetInteger(POSITION_MAGIC) == MagicNumber) {
    PositionModify(ticket, PositionGetDouble(POSITION_SL), new_tp_price);
   }
  }
 }
}

void OpenPosition(ENUM_ORDER_TYPE orderType, double lot, bool isRecoveryTrade) {
 MqlTradeRequest request;
 MqlTradeResult result;
 ZeroMemory(request);
 
 double stopLoss = 0;
 double takeProfit = 0;
 
 double point = SymbolInfoDouble(Symbol(), SYMBOL_POINT);
 
 int tpPips = isRecoveryTrade ? RecoveryTP_Pips : TP_Pips;
 
 if (orderType == ORDER_TYPE_BUY) {
  stopLoss = SymbolInfoDouble(Symbol(), SYMBOL_ASK) - SL_Pips * point;
  takeProfit = SymbolInfoDouble(Symbol(), SYMBOL_ASK) + tpPips * point;
 } else if (orderType == ORDER_TYPE_SELL) {
  stopLoss = SymbolInfoDouble(Symbol(), SYMBOL_BID) + SL_Pips * point;
  takeProfit = SymbolInfoDouble(Symbol(), SYMBOL_BID) - tpPips * point;
 }
 
 request.action = TRADE_ACTION_DEAL;
 request.symbol = _Symbol;
 request.volume = lot;
 request.type = orderType;
 request.price = SymbolInfoDouble(_Symbol, (orderType == ORDER_TYPE_BUY ? SYMBOL_ASK : SYMBOL_BID));
 request.sl = stopLoss;
 request.tp = takeProfit;
 request.deviation = 50;
 request.magic = MagicNumber;
 
 if (!OrderSend(request, result)) PrintFormat("OrderSend failed, error: %d", GetLastError());
}
//+------------------------------------------------------------------+
//| Count open positions with a specific magic number and type       |
//+------------------------------------------------------------------+
int OpenPositionsCount(int magic, ENUM_ORDER_TYPE type) {
 int count = 0;
 for (int i = 0; i < PositionsTotal(); i++) {
  if (PositionSelectByIndex(i) && PositionGetInteger(POSITION_MAGIC) == magic && PositionGetInteger(POSITION_TYPE) == type) count++;
 }
 return count;
}

bool IsNewBar() {
 static datetime lastBarTime = 0;
 datetime newBarTime = iTime(Symbol(), tf, 0);
 if (newBarTime > lastBarTime) {
  lastBarTime = newBarTime;
  return true;
 }
 return false;
}

double GetTradeLotSize(double plusDI, double minusDI) {
 if (plusDI > ADX_Lot_Increase_Threshold || minusDI > ADX_Lot_Increase_Threshold) return IncreasedLotSize;
 
 return LotSize;
}

void ManageTrailingStop() {
 for (int i = PositionsTotal() - 1; i >= 0; i--) {
  if (!PositionSelectByIndex(i) || PositionGetInteger(POSITION_MAGIC) != MagicNumber) continue;
  
  // Get the current and previous candles
  MqlRates rates[2];
  if (CopyRates(Symbol(), tf, 0, 2, rates) <= 0) continue;
  
  ulong ticket = PositionGetTicket(i);
  double sl_price = PositionGetDouble(POSITION_SL);
  double open_price = PositionGetDouble(POSITION_PRICE_OPEN);
  
  double point = SymbolInfoDouble(Symbol(), SYMBOL_POINT);
  double last_bar_open = rates[0].open;
  
  if (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) {
   // Check if the highest high has been taken out
   if (SymbolInfoDouble(_Symbol, SYMBOL_BID) > rates[1].high) {
    // This happens on a bullish candle close
    double candle_size = (rates[1].close - rates[1].open) / point;
    
    double new_sl = rates[1].low;
    
    if (candle_size >= VeryLongCandlePips) new_sl = open_price + (rates[0].close - open_price) * 0.40;
    else if (candle_size >= LongCandlePips) new_sl = last_bar_open - CandleCloseSL * point;
    
    // Only modify if the new SL is higher than the current one
    if (new_sl > sl_price + TSL_MinPips * point) PositionModify(ticket, new_sl, PositionGetDouble(POSITION_TP));
   }
  } else if (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL) {
   // Check if the lowest low has been taken out
   if (SymbolInfoDouble(Symbol(), SYMBOL_ASK) < rates[1].low) {
    // This happens on a bearish candle close
    double candle_size = (rates[0].open - rates[0].close) / point;
    
    double new_sl = rates[1].high;
    
    if (candle_size >= VeryLongCandlePips) new_sl = open_price - (open_price - rates[0].close) * 0.40;
    else if (candle_size >= LongCandlePips) new_sl = last_bar_open + CandleCloseSL * point;
    
    // Only modify if the new SL is lower than the current one
    if (new_sl < sl_price - TSL_MinPips * point) PositionModify(ticket, new_sl, PositionGetDouble(POSITION_TP));
   }
  }
 }
}


void OnTick() {
 //--- Entry Logic ---
 if (PositionsTotal() == 0)
 {
  double currentLot = GetTradeLotSize(plusDI, minusDI);
  
  if (adxValue >= ADX_Threshold)
  {
   // --- Buy Signal ---
   if (plusDI > minusDI)
   {
    OpenPosition(ORDER_TYPE_BUY, currentLot, false);
    lastTradeType = 0;
   }
   // --- Sell Signal ---
   else if (minusDI > plusDI)
   {
    OpenPosition(ORDER_TYPE_SELL, currentLot, false);
    lastTradeType = 1;
   }
  }
 }
 else
 {
  // Manage recovery trades on signal switch, regardless of ADX value
  if (plusDI > minusDI && lastTradeType == 1 && OpenPositionsCount(MagicNumber, ORDER_TYPE_SELL) > 0)
  {
   ManageRecoveryTrades(ORDER_TYPE_SELL);
  }
  else if (minusDI > plusDI && lastTradeType == 0 && OpenPositionsCount(MagicNumber, ORDER_TYPE_BUY) > 0)
  {
   ManageRecoveryTrades(ORDER_TYPE_BUY);
  }
 }
 
 // Close all trades if no recovery is active and ADX is below threshold
 if (adxValue < ADX_Threshold && GetOpenPositionsByMagic(MagicNumber, ORDER_TYPE_SELL) == 0 && GetOpenPositionsByMagic(MagicNumber, ORDER_TYPE_BUY) == 0)
 {
  CloseAllPositions();
 }
}