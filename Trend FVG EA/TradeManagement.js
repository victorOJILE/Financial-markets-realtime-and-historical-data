// Trade manager
#pragma once

bool EnterTrade(double preSL, int direction) {

 try {
  // Create a request
  MqlTradeRequest request;
  request.action = TRADE_OPERATION_REQUEST;
  request.symbol = _Symbol;
  request.deviation = 5; // Slippage
  request.magic = 123456; // Magic number
  request.comment = "My EA";
  
  double sl;
  
  if (direction == 0) {
   sl = Getstoploss(preSL, true);
   request.price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   request.type = OP_BUY;
   request.volume = CalculateLotSize(sl, true);
   request.sl = sl;
   request.tp = zoneManager.GetTakeProfit(true);
  } else {
   sl = Getstoploss(preSL, false);
   request.price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   request.type = OP_SELL;
   request.volume = CalculateLotSize(sl, false);
   request.sl = sl;
   request.tp = zoneManager.GetTakeProfit(false);
  }
  
  // Send the request
  MqlTradeResult result;
  if (!OrderSend(request, result) || result.retcode != TRADE_RETCODE_DONE) {
   Print("Error entering trade: ", GetLastError());
   return false;
  }
 } catch (e) {
  Print("Error entering trade: ", GetLastError());
 }
}

bool ModifyTrade(int ticket, double sl) {
 MqlTradeRequest request;
 request.action = TRADE_OPERATION_MODIFY;
 request.order = ticket;
 request.sl = sl;
 // Send the request
 MqlTradeResult result;
 if (!OrderSend(request, result) || result.retcode != TRADE_RETCODE_DONE) {
  Print("Trade modification failed: ", GetLastError());
  return false;
 }
}

bool CloseTrade(int ticket) {
 MqlTradeRequest request;
 request.action = TRADE_OPERATION_REQUEST;
 request.order = ticket;
 
 // Send the request
 MqlTradeResult result;
 if (!OrderSend(request, result)) {
  Print("Error closing trade: ", GetLastError());
  return false;
 }
 
 // Check the result
 if (result.retcode != TRADE_RETCODE_DONE) {
  Print("Trade closure failed: ", result.retcode);
  return false;
 }
 
 return true;
}

double Getstoploss(double preSL, bool type) {
 return type ? OB.LowerBound : OB.UpperBound;
}

double GetPips(double open, double close, bool type) {
 open = open || (type ? ASK : BID);
 return MathAbs((open - close) / _Point);
}

double CalculateLotSize(double sl, bool type) {
 double pnl = (ACCOUNT_EQUITY > ACCOUNT_BALANCE ? ACCOUNT_BALANCE : ACCOUNT_EQUITY) * RiskPerTrade / 100;
 double pips = GetPips(false, sl, type);
 
 //double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TICK_VALUE);
 double lotSize = pnl / (pips * 10);
 /*
 double lotSize = pnl / (pips * tickValue * _Point / SymbolInfoDouble(_Symbol, SYMBOL_TICK_SIZE));

 // Normalize lot size to nearest valid lot size
 double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
 double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
 double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);

 lotSize = MathRound(lotSize / lotStep) * lotStep;
 lotSize = MathMax(minLot, MathMin(maxLot, lotSize));
*/
 return NormalizeDouble(lotSize, (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS));
}