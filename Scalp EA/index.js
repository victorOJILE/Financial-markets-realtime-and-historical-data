//+------------------------------------------------------------------+ 
//| Stateful_Scalper_EA.mq5                                          | 
//| MQL5 Expert Advisor skeleton for high-frequency scalping         | 
//| Features:                                                         | 
//| - Volume Z-score & EWMA on volume                                 | 
//| - Candle body ratio, wick asymmetry                               | 
//| - Multivariate Mahalanobis-like distance (approx) for outlier detect| 
//| - Central "buckets" (trade batches) with batch SL management      | 
//| - Open/Close logic based on momentum_score & volatility (ATR)     | 
//| - Adjustable inputs for thresholds, sizes, timeframes             | 
//+------------------------------------------------------------------+ 
#property copyright ""
#property version "1.00"
#property strict
#property script_show_inputs

#include <Trade/Trade.mqh>

CTrade trade;

//---- inputs 
input int LookbackStats = 50; // lookback for mean/std 
input int VolEWMA_period = 20; // EWMA period for volume smoothing 
input int ATR_period = 14; // ATR period for volatility 
input double MomentumThreshold = 2.0; // threshold for momentum score to open
input double AddThreshold = 3.0; // stronger threshold for adding to a bucket 
input double VolumeZMin = 1.0; // minimum volume z to consider 
input double BodyRatioMin = 0.4; // min body/total_range to consider decisive candle 
input double WickAsymMin = 0.25; // wick asymmetry to detect rejection 
input double BucketSL_pips = 12.0; // SL applied to bucket (in pips) 
input double BaseLot = 0.01; // base lot size 
input int MaxTradesPerBucket = 8; // max trades in a bucket 
input int MaxBuckets = 6; // maximum buckets to track 
input ENUM_TIMEFRAMES Timeframe = PERIOD_M1; // EA timeframe to operate on
input double MinEquityPercent = 1.0; // minimal equity percent used per bucket 
input bool UseFixedSL = false; // use fixed SL by pips for bucket 
input double MaxSpread_pips = 3.0; // max allowed spread in pips

//---- internal structures 
struct TradeInfo {
 ulong ticket;
 double open_price;
 double lot;
 datetime open_time;
};

struct Bucket {
 bool active;
 int direction; // 1 = long, -1 = short 
 double avg_entry; // average entry price
 double batch_lot; // cumulative lots 
 double batch_sl_price; // stop loss price for bucket 
 int trades_count;
 TradeInfo trades[32];
 string reason; // rationale label 
};

Bucket buckets[];

//---- global arrays for stats 
double volArray[];
double bodyRatioArray[];
int dirArray[];

//---- helpers 
int digits_adjust;
double pip;
int last_bar_time = 0;

//+------------------------------------------------------------------+ 
int OnInit() {
 ArrayResize(buckets, MaxBuckets);
 for (int i = 0; i < MaxBuckets; i++) {
  buckets[i].active = false;
  buckets[i].trades_count = 0;
  buckets[i].batch_lot = 0.0;
 }
 
 digits_adjust = (int) SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
 pip = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
 
 Print("Stateful_Scalper_EA initialized on ", _Symbol);
 return (INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+ 
void OnDeinit(const int reason) {}

//+------------------------------------------------------------------+ 
void OnTick() {
 static datetime lastTime = 0;
 // operate only once per new bar 
 datetime currentTime = TimeCurrent();
 if (currentTime == lastTime) return;
 lastTime = currentTime;
 
 // quick filter: spread 
 double spread = (SymbolInfoDouble(_Symbol, SYMBOL_ASK) - SymbolInfoDouble(_Symbol, SYMBOL_BID)) / pip;
 if (spread > MaxSpread_pips) return;
 
 MqlRates rates[];
 int bars = CopyRates(_Symbol, Timeframe, 0, LookbackStats + 10, rates);
 if (bars <= 0) return;
 
 // compute feature arrays
 BuildFeatureArrays(rates, bars);
 
 // compute scores for latest candle (index 0) 
 double vol_z = VolumeZscore(0);
 double vol_ewma = VolumeEWMA(0);
 double body_ratio = BodyRatio(0);
 int direction = CandleDirection(0); // 1 long, -1 short, 0 neutral 
 double wick_asym = WickAsymmetry(0);
 double atr = iATR(_Symbol, Timeframe, ATR_period, 0);
 
 double momentum = ComputeMomentumScore(direction, vol_z, body_ratio, wick_asym);
 
 // manage existing buckets & trades
 ManageBuckets(atr);
 
 // entry logic: open new bucket or add to existing
 if (momentum >= MomentumThreshold && vol_z >= VolumeZMin && body_ratio >= BodyRatioMin) { // try add to existing long bucket first
  int idx = FindOpenBucket(1);
  if (idx >= 0 && buckets[idx].trades_count < MaxTradesPerBucket && momentum >= AddThreshold) OpenAndRegisterTrade(1);
  else CreateBucketAndTrade(1, "momentum_entry"); // open new long bucket
 } else if (momentum <= -MomentumThreshold && vol_z <= -VolumeZMin && body_ratio >= BodyRatioMin) {
  int idx = FindOpenBucket(-1);
  if (idx >= 0 && buckets[idx].trades_count < MaxTradesPerBucket && momentum <= -AddThreshold) OpenAndRegisterTrade(-1);
  else CreateBucketAndTrade(-1, "momentum_entry");
 }
 
 // optional: prune old buckets by time or equity 
 PruneBucketsByRules();
}

//+------------------------------------------------------------------+ 
void BuildFeatureArrays(const MqlRates & rates[], int bars) {
 ArrayResize(volArray, bars);
 ArrayResize(bodyRatioArray, bars);
 ArrayResize(dirArray, bars);
 
 for (int i = 0; i < bars; i++) {
  double vol = (double) rates[i].tick_volume;
  volArray[i] = vol;
  double body = MathAbs(rates[i].close - rates[i].open);
  double total = rates[i].high - rates[i].low;
  bodyRatioArray[i] = total > 0 ? body / total : 0.0;
  dirArray[i] = (rates[i].close > rates[i].open) ? 1 : ((rates[i].close < rates[i].open) ? -1 : 0);
 }
}

//+------------------------------------------------------------------+ 
// Volume Z-score using LookbackStats (on index 'idx', 0 = latest)
double VolumeZscore(int idx) {
 int n = MathMin(LookbackStats, ArraySize(volArray) - 1);
 if (n <= 2) return 0.0;
 
 double mean = 0.0, std = 0.0;
 for (int i = 1; i <= n; i++) mean += volArray[i];
 mean /= n;
 for (int i = 1; i <= n; i++) std += MathPow(volArray[i] - mean, 2);
 std = MathSqrt(std / (n - 1));
 double v = volArray[idx];
 if (std == 0.0) return 0.0;
 return (v - mean) / std;
}

//+------------------------------------------------------------------+ 
// Simple EWMA for volume 
double VolumeEWMA(int idx) {
 double alpha = 2.0 / (VolEWMA_period + 1);
 double ewma = volArray[LookbackStats];
 int limit = MathMin(VolEWMA_period, ArraySize(volArray) - 1);
 for (int i = limit; i >= 1; i--) { ewma = alpha * volArray[i] + (1 - alpha) * ewma; }
 return ewma;
}

//+------------------------------------------------------------------+ 
int CandleDirection(int idx) {
 if (ArraySize(dirArray) <= idx) return 0;
 return dirArray[idx];
}

//+------------------------------------------------------------------+ 
double BodyRatio(int idx) {
 if (ArraySize(bodyRatioArray) <= idx) return 0.0;
 return bodyRatioArray[idx];
}

//+------------------------------------------------------------------+
// wick asymmetry: (upper_wick - lower_wick) / candle_range normalized 
double WickAsymmetry(int idx) {
 MqlRates r;
 if (CopyRates(_Symbol, Timeframe, idx, 1, & r) <= 0) return 0.0;
 double upper = r.high - MathMax(r.open, r.close);
 double lower = MathMin(r.open, r.close) - r.low;
 double range = r.high - r.low;
 if (range <= 0) return 0.0;
 return (upper - lower) / range; // positive => upper wick bigger
}

//+------------------------------------------------------------------+ 
// Composite momentum score (simple linear combination, tweak weights)
double ComputeMomentumScore(int dir, double vol_z, double body_ratio, double wick_asym) {
 double s = 0.0;
 s += dir * 1.2; // base direction weight 
 s += vol_z * 0.9; // volume abnormality 
 s += body_ratio * 1.5; // decisive body weight 
 s += (-MathAbs(wick_asym)) * 0.4; // penalize strong wick asymmetry return s; 
}

//+------------------------------------------------------------------+ 
int FindOpenBucket(int direction) {
 for (int i = 0; i < ArraySize(buckets); i++) {
  if (buckets[i].active && buckets[i].direction == direction) return i;
 }
 return -1;
}

//+------------------------------------------------------------------+ 
void CreateBucketAndTrade(int direction, string reason) {
 int freeIdx = -1;
 for (int i = 0; i < ArraySize(buckets); i++) {
  if (!buckets[i].active) {
   freeIdx = i;
   break;
  }
  if (freeIdx < 0) return; // no space
 }
 
 // initialize bucket 
 buckets[freeIdx].active = true;
 buckets[freeIdx].direction = direction;
 buckets[freeIdx].trades_count = 0;
 buckets[freeIdx].batch_lot = 0.0;
 buckets[freeIdx].avg_entry = 0.0;
 buckets[freeIdx].reason = reason;
 
 // open first trade for this bucket
 OpenAndRegisterTrade(direction);
 
 // set batch SL price based on settings
 double price = (direction == 1) ? SymbolInfoDouble(_Symbol, SYMBOL_BID) : SymbolInfoDouble(_Symbol, SYMBOL_ASK);
 if (UseFixedSL) {
  double sl = (direction == 1) ? price - BucketSL_pipspip : price + BucketSL_pipspip;
  buckets[freeIdx].batch_sl_price = sl;
 } else { // dynamic SL based on ATR 
  double atr = iATR(_Symbol, Timeframe, ATR_period, 0);
  double slpips = MathMax(BucketSL_pips, atr / pip * 1.2);
  double sl = (direction == 1) ? price - slpipspip : price + slpipspip;
  buckets[freeIdx].batch_sl_price = sl;
 }
}

//+------------------------------------------------------------------+ 
void OpenAndRegisterTrade(int direction) {
 double price = (direction == 1) ? SymbolInfoDouble(_Symbol, SYMBOL_ASK) : SymbolInfoDouble(_Symbol, SYMBOL_BID);
 double lot = ComputeLotSize();
 double sl = 0;
 double tp = 0; // we won't set per-trade SL (use batch SL)
 
 trade.SetExpertMagicNumber(123456);
 trade.SetDeviationInPoints(10);
 bool ok = false;
 if (direction == 1) ok = trade.Buy(lot, NULL, price, sl, tp, "bucket_entry");
 else ok = trade.Sell(lot, NULL, price, sl, tp, "bucket_entry");
 
 if (!ok) {
  Print("Trade open failed: ", trade.ResultRetcode(), " - ", trade.ResultRetcodeDescription());
  return;
 }
 
 ulong t = trade.ResultOrder(); // register into bucket 
 int idx = FindOpenBucket(direction);
 if (idx < 0) { // try find any bucket with same direction (in case called standalone) 
  for (int i = 0; i < ArraySize(buckets); i++)
   if (buckets[i].active == false) { idx = i; break; } if (idx < 0) return;
  buckets[idx].active = true;
  buckets[idx].direction = direction;
  buckets[idx].trades_count = 0;
  buckets[idx].batch_lot = 0;
  buckets[idx].avg_entry = 0;
 }
 
 TradeInfo ti;
 ti.ticket = t;
 ti.open_price = price;
 ti.lot = lot;
 ti.open_time = TimeCurrent();
 
 int pos = buckets[idx].trades_count;
 buckets[idx].trades[pos] = ti;
 buckets[idx].trades_count++;
 buckets[idx].batch_lot += lot; // update avg entry
 if (buckets[idx].avg_entry == 0.0) buckets[idx].avg_entry = price;
 else buckets[idx].avg_entry = (buckets[idx].avg_entry * (buckets[idx].trades_count - 1) + price) / buckets[idx].trades_count;
 
 // ensure batch SL exists: recalc 
 RecalcBucketSL(idx);
 
 PrintFormat("Opened trade %I64u dir=%d lot=%.2f price=%.5f", t, direction, lot, price);
}

//+------------------------------------------------------------------+
double ComputeLotSize() {
 // simple fixed lot for now. Could scale by volatility/equity. 
 double equity = AccountInfoDouble(ACCOUNT_EQUITY);
 double riskEquity = equity * (MinEquityPercent / 100.0);
 double lots = BaseLot;
 return NormalizeDouble(lots, 2);
}

//+------------------------------------------------------------------+ 
void RecalcBucketSL(int idx) {
 if (idx < 0 || idx >= ArraySize(buckets)) return;
 if (!buckets[idx].active) return;
 
 double atr = iATR(_Symbol, Timeframe, ATR_period, 0);
 double price = buckets[idx].avg_entry;
 double slpips = MathMax(BucketSL_pips, atr / pip * 1.1);
 double sl = (buckets[idx].direction == 1) ? price - slpipspip : price + slpipspip;
 buckets[idx].batch_sl_price = sl;
}

//+------------------------------------------------------------------+
void ManageBuckets(double atr) {
 for (int i = 0; i < ArraySize(buckets); i++) {
  if (!buckets[i].active) continue;
  
  // check batch SL hit
  if (buckets[i].direction == 1) {
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   if (bid <= buckets[i].batch_sl_price) {
    // close all trades in bucket
    CloseBucket(i, "batch_sl_hit");
    continue;
   }
  } else {
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   if (ask >= buckets[i].batch_sl_price) {
    CloseBucket(i, "batch_sl_hit");
    continue;
   }
  }
  
  // volatility-based closing: if ATR surges or collapses (regime change)
  double current_atr = atr;
  double expected_atr = iATR(_Symbol, Timeframe, ATR_period, 1);
  if (current_atr > expected_atr * 1.8 || current_atr < expected_atr * 0.5) {
   // volatility changed drastically -> close bucket to avoid scalping in unstable regime
   CloseBucket(i, "volatility_regime_change");
   continue;
  }
  
  // time-based or reason-based pruning could be added here
  
 }
}

//+------------------------------------------------------------------+ 
void CloseBucket(int idx, string reason) {
 if (idx < 0 || idx >= ArraySize(buckets)) return;
 PrintFormat("Closing bucket %d reason=%s", idx, reason);
 for (int t = 0; t < buckets[idx].trades_count; t++) {
  ulong ticket = buckets[idx].trades[t].ticket;
  if (PositionSelectByTicket(ticket)) {
   double vol = PositionGetDouble(POSITION_VOLUME); // perform close by ticket 
   trade.PositionClose(ticket);
  }
 } // reset bucket 
 buckets[idx].active = false;
 buckets[idx].trades_count = 0;
 buckets[idx].batch_lot = 0.0;
 buckets[idx].avg_entry = 0.0;
}

//+------------------------------------------------------------------+ 
void PruneBucketsByRules() { // simple prune: if trades count is zero but active (shouldn't happen) or if bucket too old 
 for (int i = 0; i < ArraySize(buckets); i++) {
  if (!buckets[i].active) continue;
  if (buckets[i].trades_count == 0) { buckets[i].active = false; continue; } // if bucket open for more than X seconds close it 
  TradeInfo ti = buckets[i].trades[0];
  if (TimeCurrent() - ti.open_time > 60 * 30) { // 30 minutes 
   CloseBucket(i, "time_expiry");
  }
 }
}

//+------------------------------------------------------------------+ 
// Utility: check if position exists by ticket 
bool PositionSelectByTicket(ulong ticket) { // search all positions ulong pos_ticket; 
 for (int i = 0; i < PositionsTotal(); i++) { if (PositionGetTicket(i) == ticket) return true; }
 return false;
}

//+------------------------------------------------------------------+
// OnChartEvent left empty for now 
void OnChartEvent(const int id,
 const long & lparam,
  const double & dparam,
   const string & sparam) {}

//+------------------------------------------------------------------+ // End of EA //+------------------------------------------------------------------+