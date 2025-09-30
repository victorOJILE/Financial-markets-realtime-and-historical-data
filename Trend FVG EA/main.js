//+------------------------------------------------------------------+
//|   Trend+FVG EA with State Machine, Fractals, BOS, and CTrade     |
//|   Generates entries from FVG after 2 BOS |
//|   Author: Victor + Assistant (GPT-5 Thinking mini)               |
//+------------------------------------------------------------------+
#property strict
#property version "1.00"
#property description "Trend+FVG EA: unmitigated swings, BOS x2, FVG pullback entries"
#property copyright "Victor"

#include <Trade\Trade.mqh>

#include <TradeManagement.mqh>

#define NUM_TF 4

input int SessionStartHour = 0;
input int SessionEndHour = 16;
input double R2_Threshold = 0.85;
input double Lots = 0.01;
input int MinBarsForScan = 120; // history bars to copy

//--- Enums and structs
enum TrendState { UP_TREND, DOWN_TREND, RANGE_STATE };
enum TradeState { INIT, SCANNING, WAITING_FOR_ENTRY, WAITING_FOR_CONFIRMATION, TRADE_ACTIVE };

struct SwingPoint {
 int index;
 double price;
};

struct FVG {
 double UpperBound;
 double LowerBound;
}

struct TrendLineResults {
 bool success;
 bool outlierDetected;
 double slope;
 double intercept;
 double R_squared;
 double clusterIdx[];
 double meanRes;
 double stdRes;
 double maxAbsRes;
};

struct BOS_Record {
 datetime timeBroken;
 double priceBroken;
 bool isHigh;      // true if swing was a high, false if low
};
//--- Structs for per-timeframe state
struct TFContext {
 ENUM_TIMEFRAMES tf;
 MqlRates rates[];
 SwingPoint highs[];
 SwingPoint lows[];
 TrendLineResults trend;
 TrendLineResults trendlines[];
 FVG fvgLow[];
 FVG fvgHigh[];
 BOS_Record lastBOS;
 int bosCount;
 TrendState TrendS;
 TradeState tradeState;
 bool entryReady;
};

struct Decision {
 int direction; // 1=BUY, -1=SELL, 0=NONE
 double entry;
 double sl;
 double tp;
 ENUM_TIMEFRAMES entryTF;
};

//--- Globals
datetime gLastBarTime = 0;
CTrade trade;

ENUM_TIMEFRAMES TF_LIST[NUM_TF] = { PERIOD_H4, PERIOD_M30, PERIOD_M15, PERIOD_M5 };
double BOS_STD_MULT = 2.0; // residual must exceed mean +- 2*std to count
double BOS_MAX_MULT = 1.5; // or exceed 1.5 * max abs cluster residual
int BOS_CONFIRM_BARS = 2;

class TrendMachine {
 public: 
 TFContext CTX[NUM_TF];
 
 TrendMachine() {
  for (int i = 0; i <= NUM_TF; i++) {
   CTX[i].tf = TF_LIST[i];
   CTX[i].tradeState = INIT;
   CTX[i].TrendS =  RANGE_STATE;
   ArraySetAsSeries(ctx.rates, true);
   ArraySetAsSeries(ctx.highs, true);
   ArraySetAsSeries(ctx.lows, true);
  }
 }
 // --- destructor
 ~TrendMachine() {
  for (int k = 0; k < NUM_TF; k++) {
   TFContext & ctx = CTX[k];
   
   ArrayFree(ctx.rates);
   ArrayFree(ctx.highs);
   ArrayFree(ctx.lows);
   ArrayFree(ctx.fvgHigh);
   ArrayFree(ctx.fvgLow);
  }
 }
 // On new bar, check if candle is an outlier to avoid unnecessary reruns
 bool ConfirmTrendValidation(TFContext ctx) {
  double price = (ctx.TrendS == UP_TREND) ? iLow(_Symbol, ctx.tf, 1) : iHigh(_Symbol, ctx.tf, 1);
  double residual = ComputeSingleResidual(ArraySize(ctx.trend.clusterIdx), price, ctx.trend);
  bool cond = false;
  if(ctx.TrendS == UP_TREND && residual < 0) cond = true;
  if(ctx.TrendS == DOWN_TREND && residual > 0) cond = true;
  
  if(!cond) {
   ctx.tradeState = SCANNING;
   Print("➡ State changed: WAITING_FOR_CONFIRMATION → SCANNING");
  }
  
  return cond;
 }
 
 void GetTrend(TFContext &ctx) {
  TrendLineResults highTrend = GetTrendLines(ctx.lows);
  TrendLineResults lowTrend = GetTrendLines(ctx.highs, ctx);
  
  switch (ctx.TrendS) {
   case UP_TREND:
    if(highTrend.success && highTrend.clusterIdx > 1) DetectBOS_ByClusterResiduals(ctx.lows, highTrend, ctx.rates);
    
    // TODO: We should implement a BoS by swings check, but
    // A swing break if used as trendline point can be detected by DetectBOS_ByClusterResiduals
    // In a downtrend, a bos by swing can happen to the upside while price is still below trend line
    
    if (highTrend.success && !highTrend.outlierDetected) {
     if(ctx.bosCount >= 1 && DetectContinuousBOS(ctx)) {
      ctx.bosCount++;
      Print("BOS Up detected at ", TimeCurrent());
     }
    } else {
     // TODO: After new 
     ctx.TrendS = RANGE_STATE;
     ctx.bosCount = 1;
     ctx.lastBOS.timeBroken = ctx.rates[1].time;
     ctx.lastBOS.priceBroken = ctx.highs[idx].price;
     ctx.lastBOS.isHigh = true;
     ctx.outlierDetected = false;
     ZeroMemory(ctx.trend);
    }
    break;
   case DOWN_TREND:
    if(lowTrend.success && lowTrend.clusterIdx > 1) DetectBOS_ByClusterResiduals(ctx.highs, lowTrend, ctx.rates);
    if (lowTrend.success && !lowTrend.outlierDetected) {
     if (ctx.bosCount >= 1 && DetectContinuousBOS(ctx)) {
      ctx.bosCount++;
      Print("BOS Down detected at ", TimeCurrent());
     }
    } else {
     ctx.TrendS = RANGE_STATE;
     ctx.bosCount = 1;
     ctx.lastBOS.timeBroken = ctx.rates[1].time;
     ctx.lastBOS.priceBroken = ctx.lows[idx].price;
     ctx.lastBOS.isHigh = false;
     ctx.outlierDetected = false;
     ZeroMemory(ctx.trend);
    }
    break;
   case RANGE_STATE:
    if (highTrend.success && highTrend.R_squared > lowTrend.R_squared) {
     ctx.state = UP_TREND;
     ctx.trend = highTrend;
     ctx.bosCount = 1;
     ctx.lastBOS.timeBroken = ctx.rates[1].time;
     ctx.lastBOS.priceBroken = ctx.highs[1].price;
     ctx.lastBOS.isHigh = true;
     ctx.outlierDetected = false;
     
     Print("Transition RANGE->UP at ", TimeCurrent());
    } else if (lowTrend.success) {
     ctx.state = DOWN_TREND;
     ctx.trend = lowTrend;
     ctx.bosCount = 1;
     ctx.lastBOS.timeBroken = ctx.rates[1].time;
     ctx.lastBOS.priceBroken = ctx.lows[1].price;
     ctx.lastBOS.isHigh = false;
     ctx.outlierDetected = false;
     
     Print("Transition RANGE->DOWN at ", TimeCurrent());
    } else {
     ctx.bosCount = 0;
     ZeroMemory(ctx.trend);
    }
    
    break;
  }
 }
 void CheckEntry() {
  GetTrend();
  if (bosCount < 2 || !entryReady) {
   state = SCANNING;
   return false;
  }
  
  double lowest, highest;
  
  if (FVG.Type == OB_BUY) {
   highest = iHighest(symbol, PERIOD_CURRENT, MODE_HIGH, FVG.Index, 0);
   lowest = iLow(symbol, PERIOD_CURRENT, iLowest(symbol, PERIOD_CURRENT, MODE_LOW, highest, 0));
   highest = iHigh(symbol, PERIOD_CURRENT, highest);
   
   if (highest > SymbolInfoDouble(symbol, SYMBOL_BID)) {
    Print(iTime(symbol, PERIOD_CURRENT, 1) + " Entry checks halted till blocks are refreshed. Price gone out of bounds!");
    checkForEntry = false;
    return false;
   }
  } else {
   lowest = iLowest(symbol, PERIOD_CURRENT, MODE_LOW, FVG.Index, 0);
   highest = iHigh(symbol, PERIOD_CURRENT, iHighest(symbol, PERIOD_CURRENT, MODE_HIGH, lowest, 0));
   lowest = iLow(symbol, PERIOD_CURRENT, lowest);
   
   if (lowest < SymbolInfoDouble(symbol, SYMBOL_BID)) {
    Print(iTime(symbol, PERIOD_CURRENT, 1) + " Entry checks halted till blocks are refreshed. Price gone out of bounds!");
    checkForEntry = false;
    return false;
   }
  }
  
  double price = FVG.Type == OB_BUY ? lowest : highest;
  
  return price <= FVG.UpperBound && price >= LowerBound;
 }
 void Scan(TFContext &ctx) {
  GetTrend(ctx);
  if (ctx.bosCount < 2 || !ctx.entryReady) break;
  ctx.tradeState = WAITING_FOR_ENTRY;
  Print("➡ State changed: SCANNING → WAITING_FOR_ENTRY");
  WAITING_FOR_ENTRY(ctx);
 }
 
 void WAITING_FOR_ENTRY(TFContext &ctx) {
  if (!CheckEntry(ctx)) break;

  if (ctx.TrendS == UP_TREND) {
   
  } else if (ctx.TrendS == DOWN_TREND) {
   
  }
  
  ctx.tradeState = WAITING_FOR_CONFIRMATION;
  Print("➡ State changed: WAITING_FOR_ENTRY → WAITING_FOR_CONFIRMATION");
 }
 void RefreshDatasets(TFContext &ctx[]) {
  ArrayResize(ctx.rates, 0);
  if (CopyRates(_Symbol, ctx.tf, 0, MinBarsForScan, ctx.rates) < MinBarsForScan) {
   Print("❌ Failed to load rates for TF=", EnumToString(ctx.tf), " Time: ", TimeCurrent());
   return;
  }
  
  ExtractLiquidityPoints(ctx);
 }
}

// TODO: We need an array state that tracks a certain process that leads to an entry signal 
// Different params should trigger different actions

// TODO: DetectTrend should be called once on init,
// After a trend line break, we need to start tracking new trend, checking only for outlier
// We get two new swings in the new trend, then set it as the new trend

// TODO: After BOS by residual, we should check if a swing was broken, to increment boscount

TrendMachine gMachine;

bool IsFlatSlope(TFContext &ctx, int swingIndex) {
 int k = 1;
 while (k < swingIndex +1) {
  k++;
  double y1 = ctx.rates[k +2].close;
  double y2 = ctx.rates[k +1].close;
  double y3 = ctx.rates[k].close;
  
  double yArr[3] = { y1, y2, y3 };
  int n = ArraySize(yArr);
  double x_mean = (n + 1) / 2.0;
  double y_mean = (y1 + y2 + y3) / n;
  
  double numerator = 0, denominator = 0;
  
  for (int i = 1; i <= n; i++) {
   numerator += (i - x_mean) * (yArr[i - 1] - y_mean);
   denominator += MathPow(i - x_mean, 2);
  }
  
  if (denominator == 0) continue;
  double slope = numerator / denominator;
  
  if((ctx.trend == UP_TREND && slope < 0) || (ctx.trend == DOWN_TREND && slope > 0)) {
   return true;
  }
 }
 return false;
}

TrendLineResults CalculateRegression(double & prices[]) {
 TrendLineResults r;
 r.R_squared = 0;
 
 int n = ArraySize(prices) - 1;
 if (n < 2) return r;
 
 double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0, sumY2 = 0;
 
 for (int i = 0; i <= n; i++) {
  double y = prices[i];
  sumX += i;
  sumY += y;
  sumXY += i * y;
  sumX2 += i * i;
  sumY2 += y * y;
 }
 
 double denom = n * sumX2 - sumX * sumX;
 if (denom == 0) return r;
 
 r.slope = (n * sumXY - sumX * sumY) / denom;
 r.intercept = (sumY - r.slope * sumX) / m;
 
 double numerator_r2 = n * sumXY - sumX * sumY;
 double denom_r2_x = n * sumX2 - sumX * sumX;
 double denom_r2_y = n * sumY2 - sumY * sumY;
 double denom_r2 = denom_r2_x * denom_r2_y;
 
 r.R_squared = (denom_r2 == 0) ? 0 : (numerator_r2 * numerator_r2) / denom_r2;
 
 r.success = r.R_squared >= R2_Threshold;
 
 return r;
}

double ComputeSingleResidual(int n, double price, TrendLineResults &trend) {
 double predicted = trend.slope * n + trend.intercept;
 
 return price - predicted;
}
void ComputeClusterResidualStats(const double &prices[], const int &clusterIdx [], TrendLineResults &trend) {
 int clusterSize = ArraySize(clusterIdx);
 if (clusterSize <= 3) {
  ctx.meanRes = ctx.stdRes = ctx.maxAbsRes = 0;
  return;
 }
 
 // residuals are computed in regression x-space: x = 0..clusterSize-1
 double sum = 0;
 double sumsq = 0;
 ctx.maxAbsRes = 0;
 
 for (int i = 0; i < clusterSize; i++) {
  int idx = clusterIdx[i]; // index into prices[]
  double res = ComputeSingleResidual(i, prices[idx].price, trend);
  
  sum += res;
  sumsq += res * res;
  if (MathAbs(res) > maxAbsRes) maxAbsRes = MathAbs(res);
 }
 
 ctx.meanRes = sum / clusterSize;
 double variance = (sumsq / clusterSize) - (ctx.meanRes * ctx.meanRes);
 if (variance < 0) variance = 0;
 ctx.stdRes = MathSqrt(variance);
}

bool DetectBOS_ByClusterResiduals(const double &prices[], TrendLineResults & trend,
 const MqlRates &rates[]) {
 // thresholds
 double th_std = BOS_STD_MULT * trend.stdRes;
 double th_max = BOS_MAX_MULT * trend.maxAbsRes;
 
 // We'll check the last N closes (rates[1] = latest close). Require BOS_CONFIRM_BARS consecutive confirms.
 int confirms = 0;
 int ncheck = MathMax(BOS_CONFIRM_BARS, 3); // check at least a few bars (safe)
 
 int pricesCount = ArraySize(prices);
 // next x for projection = pricesCount (one step beyond last reg point)
 // subsequent bars -> x = pricesCount + offset
 for (int offset = 0; offset < ncheck && offset < ArraySize(rates); offset++) {
  // use the close price of the market for test (rates[ offset ].close since series is set as series)
  double residual = rates[offset].close - (trend.slope * (pricesCount + offset) + trend.intercept);
  
  // Uptrend: BOS is bearish move (residual strongly negative)
  if (trend.slope > 0) {
   if (residual < (trend.meanRes - MathMax(th_std, th_max))) confirms++; // strong negative residual
   else confirms = 0;
  } else { // Downtrend: BOS is bullish move (residual strongly positive)
   if (residual > (trend.meanRes + MathMax(th_std, th_max))) confirms++; // strong positive residual
   else confirms = 0;
  }
  
  if (confirms >= BOS_CONFIRM_BARS) {
   trend.outlierDetected = true;
   return true;
  }
 }
 
 return false;
}

bool DetectContinuousBOS(TFContext &ctx) {
 // --- continuation BOS case (flat slope + break)
 if (ctx.TrendS == UP_TREND) {
  int idx = ArraySize(ctx.highs) - 1;
  // We need to check for a continuous bos first, then we can run IsFlatSlope and DetectContinuousBOS as final check
  if (ctx.rates[1].close > ctx.highs[idx].price && IsFlatSlope(ctx, ctx.highs[idx].index)) {
   ctx.lastBOS.timeBroken = rates[1].time;
   ctx.lastBOS.priceBroken = ctx.highs[idx].price;
   ctx.lastBOS.isHigh = true;
   return true;
  }
 } else if(ctx.TrendS == DOWN_TREND) {
  int idx = ArraySize(ctx.lows) - 1;
  if (ctx.rates[1].close < ctx.lows[idx].price && IsFlatSlope(ctx, ctx.highs[idx].index)) {
   ctx.lastBOS.timeBroken = rates[1].time;
   ctx.lastBOS.priceBroken = ctx.lows[idx].price;
   ctx.lastBOS.isHigh = false;
   return true;
  }
 }
 
 return false;
}

TrendLineResults GetTrendLines(SwingPoint & prices[], TFContext &ctx) {
 for(int k = ArraySize(ctx.trendlines); k <= 0; k--) {
  if(ctx.trend.slope > 0) { // uptrend
   int res = ComputeSingleResidual(ArraySize(ctx.trend.clusterIdx), iLow(_Symbol, ctx.tf, 1), ctx.trend);
   if(res > ctx.trend.meanRes) ArrayResize(ctx.trendlines, ArraySize(ctx.trendlines) -1);
  } else {
   
  }
 }
 if(true) return ctx.trend;
 
 ArrayResize(ctx.trendlines, 0);
 
 TrendLineResults best;
 best.success = false;
 best.R_squared = 0;
 
 int startSwingCount = 1;
 int n = ArraySize(prices);
 if (n < startSwingCount) return best;
 
 int finalCluster[];
 
 for(int k = startSwingCount; k < n; k++) {
  // Work array to filter indexes
  int idx[];
  for(int i = 0; i <= k; i++) idx[i] = i;
 
  while (ArraySize(idx) > 3) {
   // Build sub-prices from current idx[]
   double subPrices[];
   for (int j = 0; j < ArraySize(idx); j++) {
    SwingPoint & p = prices[idx[j]];
    subPrices[j] = p.price;
   }
   
   TrendLineResults r = CalculateRegression(subPrices);
   if (!r.success) break;
   
   // Filter residuals by slope direction
   int newIdx[];
   for (int j = 0; j < ArraySize(idx); j++) {
    double residual = ComputeSingleResidual(j, subPrices[j], r);
    
    // Uptrend → keep BELOW line (negative residuals)
    // Downtrend → keep ABOVE line (positive residuals)
    if ((r.slope > 0 && residual <= 0) || (r.slope < 0 && residual >= 0)) newIdx[ArraySize(newIdx)] = idx[j];
   }
   // Store if better
   if (r.R_squared > best.R_squared) {
    r.success = true;
    best = r;
    ArrayCopy(finalCluster, newIdx);
   }
   
   // Update working set
   idx = newIdx;
  }
 }
 
 ComputeClusterResidualStats(prices, finalCluster, trend);
 ArrayCopy(best.clusterIdx, finalCluster);
 
 return best;
}
// Get trend 0 - bullish, 1 - bearish, -1 none
int DetectTrend(SwingPoint & gHs[], SwingPoint & gLs[], double threshold) {
 int nHighs = ArraySize(gHs);
 int nLows = ArraySize(gLs);
 int maxIndex = MathMax(nHighs, nLows);
 
 int scoreHighs = 0;
 int scoreLows = 0;
 
 for (int i = 0; i < maxIndex; i++) {
  if (i < nHighs) scoreHighs++;
  if (i < nLows) scoreLows++;
  
  int total = scoreHighs + scoreLows;
  if (total < 5) continue;
  
  double ratioHighs = (double) scoreHighs / total;
  double ratioLows = (double) scoreLows / total;
  
  // Decision at this window size
  if (ratioHighs >= threshold) return 1;
  if (ratioLows >= threshold) return 0;
 }
 
 return -1; // no confirmation
}
// Extracts unmitigated highs and lows
void ExtractLiquidityPoints(TFContext &ctx) {
 SwingPoint &gHs = ctx.highs;
 SwingPoint &gLs = ctx.lows;
 MqlRates &rates[] = ctx.rates;
 
 ArrayResize(gHs, 0);
 ArrayResize(gLs, 0);
 
 int curHigh = rates[1].high;
 int curLow = rates[1].low;
 int dir = IsBullish(rates[1]);
 
 // Starting from current candle
 for (int i = 1; i < MinBarsForScan; i++) {
  double hi = rates[i].high;
  double lo = rates[i].low;
  bool curDir = IsBullish(rates[i]);
  
  // --- High point
  if (hi >= curHigh) {
   int sz = ArraySize(gHs);
   if (curDir == dir) { // update last swing
    SwingPoint & prev = gHs[sz - 1];
    prev.index = i;
    prev.price = hi;
   } else {
    dir = curDir;
    SwingPoint h = { i, hi };
    gHs[sz] = h;
   }
   
   curHigh = hi;
  }
  
  // --- Low point
  if (lo >= curLow) continue;
  if (curDir == dir) { // update last swing
   SwingPoint & prev = gLs[ArraySize(lows) - 1];
   prev.index = i;
   prev.price = lo;
  } else {
   dir = curDir;
   SwingPoint l = { i, lo };
   gLs[ArraySize(lows)] = l;
  }
  
  curLow = lo;
 }
}

bool IsBullish(MqlRates & rate) {
 return rate.close > rate.open;
}

//| FVG detection loop                                       |
//| direction: 0 bullish, 1 bearish                                |
bool DetectFVGs(TFContext &ctx) {
 int weakOppositeCount = 0;
 int consecutiveOpposite = 0;
 bool fvgFound = false;
 SwingPoint & arr = direction == 1 gMachine.unmitigatedHighs: gMachine.unmitigatedLows;
 
 for (int i = 0; i < MinBarsForScan; i++) {
  fvgFound = direction == 1 ? rates[i].high < rates[i + 2].low : rates[i].low > rates[i + 2].high;
  if (fvgFound) break;
  // TODO: Error here
  // opposite candle check
  bool opposite = direction == 0 ? rates[i].close < rates[i].open : rates[i].close > rates[i].open;
  
  if (opposite) {
   consecutiveOpposite++;
   weakOppositeCount++;
   if (weakOppositeCount > 1 || consecutiveOpposite >= 2) break;
  } else consecutiveOpposite = 0;
 }
 
 return fvgFound;
}

void CloseAllPositions() {
 for (int i = PositionsTotal() - 1; i >= 0; i--) {
  ulong ticket = PositionGetTicket(i);
  if (PositionSelectByTicket(ticket)) {
   // string sym = PositionGetString(POSITION_SYMBOL);
   // double vol = PositionGetDouble(POSITION_VOLUME);
   long type = (long) PositionGetInteger(POSITION_TYPE);
   if (type == POSITION_TYPE_BUY) trade.PositionClose(ticket);
   if (type == POSITION_TYPE_SELL) trade.PositionClose(ticket);
  }
 }
}

int OnInit() {
 Print("TrendFVG EA initialized.");
 return INIT_SUCCEEDED;
}

void OnTick() {
 // only on new bar close
 datetime curBarTime = TimeCurrent();
 if (curBarTime == gLastBarTime) return;
 gLastBarTime = curBarTime;
 
 if (Bars(_Symbol, _Period) < MinBarsForScan) return;
 
 // session filter - server time 
 MqlDateTime dt;
 TimeToStruct(now, dt);
 //if (SessionStartHour <= SessionEndHour && !(dt.hour >= SessionStartHour && dt.hour < SessionEndHour)) return; // wrap-around (e.g., start 20, end 6)
 if (!(dt.hour >= SessionStartHour || dt.hour < SessionEndHour)) return;
 
 //datetime currentTime = iTime(_Symbol, ctx.tf, 1);
 for (int k = 0; k < NUM_TF; k++) {
  TFContext & ctx = gMachine.CTX[k];
  // Skip multiple processing for same time on HTFs
  //if (ctx.rates[1].time == currentTime) continue;
  
  switch (ctx.tradeState) {
   case INIT: {
    RefreshDatasets(ctx);
    ctx.tradeState = SCANNING;
    Print("➡ State changed: INIT → SCANNING");
    int trend = DetectTrend(ctx.highs, ctx.lows, 0.6);
    if(trend == 0) ctx.TrendS = UP_TREND;
    else ctx.TrendS = DOWN_TREND;
    
    gMachine.Scan(ctx);
    break;
   }
   case SCANNING:
    gMachine.Scan(ctx);
    break;
   case WAITING_FOR_ENTRY:
    if (!gMachine.ConfirmTrendValidation(ctx)) {
     ctx.tradeState = SCANNING;
     Print("➡ Trend Invalidated: WAITING_FOR_ENTRY → SCANNING");
     gMachine.Scan(ctx);
    }
    gMachine.WAITING_FOR_ENTRY(ctx);
    break;
   case WAITING_FOR_CONFIRMATION:
    if (!gMachine.ConfirmTrendValidation(ctx)) {
     ctx.tradeState = SCANNING;
     Print("➡ Entry Invalidated: WAITING_FOR_CONFIRMATION → SCANNING");
     gMachine.Scan(ctx);
    }
    
    if (!gMachine.CheckCandleSticksComfirmation(ctx)) break;
    
    if (ctx.TrendS == UP_TREND) {
     EnterTrade(0, ctx);
    } else if (ctx.TrendS == DOWN_TREND) {
     EnterTrade(1, ctx);
    }
    
    ctx.tradeState = TRADE_ACTIVE;
    Print("➡ State changed: WAITING_FOR_CONFIRMATION → TRADE_ACTIVE")
    break;
   case TRADE_ACTIVE:
    if (!ManageTrade(ctx)) {
     ctx.tradeState = SCANNING;
     Print("➡ State changed: TRADE → SCANNING");
    }
    break;
   default:
    ctx.tradeState = INIT;
    break;
  }
  
  gMachine.RefreshDatasets(ctx);
 }
}
//------------------------------------------------------------ 
// Fuse TF signals: weighted voting
//------------------------------------------------------------
Decision ProcessTopDownDecision(string symbol) {
 int weights[NUM_TF] = { 3, 2, 1, 1 }; // example: H4=3, H1=2, M15=1, M5=1 
 double score = 0;
 for (int k = 0; k < NUM_TF; k++) { score += contexts[k].state * weights[k]; }
 
 Decision d;
 d.direction = 0;
 d.entry = 0;
 d.sl = 0;
 d.tp = 0;
 d.entryTF = PERIOD_CURRENT;
 
 if (score >= 2.0) d.direction = 1;
 else if (score <= -2.0) d.direction = -1;
 
 // pick lowest TF aligned with decision 
 if (d.direction != 0) {
  for (int k = NUM_TF - 1; k >= 0; k--) {
   if (contexts[k].state == d.direction && ArraySize(contexts[k].fvgLow) > 0) {
    d.entryTF = contexts[k].tf;
    int last = ArraySize(contexts[k].fvgLow) - 1;
    d.entry = (contexts[k].fvgLow[last] + contexts[k].fvgHigh[last]) / 2;
    d.sl = (d.direction == 1 ? contexts[k].lows[0].price : contexts[k].highs[0].price);
    d.tp = d.entry + d.direction2(MathAbs(d.entry - d.sl));
    break;
   }
  }
 }
 return d;
}

//------------------------------------------------------------ 
// Run engine each bar
//------------------------------------------------------------ 
void RunMultiTFEngine(string symbol, int barsBack = 300) {
 BuildTFContexts(symbol, barsBack);
 Decision d = ProcessTopDownDecision(symbol);
 if (d.direction == 1) PlaceBuy(symbol, d.entry, d.sl, d.tp);
 if (d.direction == -1) PlaceSell(symbol, d.entry, d.sl, d.tp);
}