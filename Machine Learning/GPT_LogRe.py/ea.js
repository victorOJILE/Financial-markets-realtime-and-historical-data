//+------------------------------------------------------------------+ 
//|                 ML Bias Reader & Trade Executor.mq5              | 
//|     Reads AI bias prediction and places a trade if none open     |
//+------------------------------------------------------------------+ 
#property strict 
input string PredictionFile = "bias_prediction.txt"; 
input double LotSize = 0.1; 
input double Slippage = 3; 
input double StopLoss = 100; 
input double TakeProfit = 200;

string lastBias = "";

void PlaceTrade(string bias) {
 if (PositionSelect(_Symbol)) return; // already have a position
 
 double price = (bias == "up") ? SymbolInfoDouble(_Symbol, SYMBOL_ASK) : SymbolInfoDouble(_Symbol, SYMBOL_BID);
 
 double sl = (bias == "up") ? price - StopLoss * _Point : price + StopLoss * _Point;
 double tp = (bias == "up") ? price + TakeProfit * _Point : price - TakeProfit * _Point;
 ENUM_ORDER_TYPE type = (bias == "up") ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
 
 MqlTradeRequest request;
 MqlTradeResult result;
 ZeroMemory(request);
 ZeroMemory(result);
 
 request.action = TRADE_ACTION_DEAL;
 request.symbol = _Symbol;
 request.volume = LotSize;
 request.type = type;
 request.price = price;
 request.sl = sl;
 request.tp = tp;
 request.deviation = Slippage;
 request.type_filling = ORDER_FILLING_FOK;
 
 if (!OrderSend(request, result) || result.retcode != TRADE_RETCODE_DONE) { Print("Trade failed: ", result.retcode); } else { Print("Trade placed: ", (bias == "up" ? "BUY" : "SELL"), " @ ", price);
  lastBias = bias; }
}

int OnInit() {
 Print("AI Bias Trader EA Initialized.");
 return INIT_SUCCEEDED;
}

void OnTick() {
 // On candle close
 ResetLastError();
 int handle = FileOpen(PredictionFile, FILE_READ | FILE_TXT | FILE_ANSI); 
 if (handle == INVALID_HANDLE) {
  Print("Failed to open prediction file: ", filename, " Error: ", GetLastError()); 
  return "";
 }
 
 string content = FileReadString(handle);
 FileClose(handle);
 string bias = StringTrim(content);

 if ((bias != "up" && bias != "down") || bias == lastBias) return; // No new prediction or already used
 
 PlaceTrade(bias);
}