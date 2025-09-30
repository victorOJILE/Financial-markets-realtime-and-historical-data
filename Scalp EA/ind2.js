//+------------------------------------------------------------------+ 
//| Stateful_Scalper_EA.mq5                                          | 
//| MQL5 Expert Advisor for high-frequency stateful scalping         | 
//| Updated features:                                                 | 
//| - Balance-based dynamic lot sizing (risk per bucket using SL)     | 
//| - Partial close logic (profit-based, multi-step partials)         | 
//| - Session trading filter (start/end hour)                         | 
//| - Volume smoothing via EWMA + simple Kalman-like smoother         | 
//| - Outlier detection using Mahalanobis-like distance on (vol,body) | 
//| - Central "buckets" with batch SL (no per-trade SL)              | 
//| - Other: spread guard, ATR-based dynamic SL, pruning & logging    |
//+------------------------------------------------------------------+
#property copyright ""
#property version "1.10"
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
input double BucketSL_pips = 12.0; // fallback SL applied to bucket (in pips) 
input double RiskPercentPerBucket = 0.5; // percent of balance risked per bucket (balance-based sizing) 
input double BaseLot = 0.01; // minimum lot
input int MaxTradesPerBucket = 8; // max trades in a bucket
input int MaxBuckets = 6; // maximum buckets to track
input ENUM_TIMEFRAMES Timeframe = PERIOD_M1; // EA timeframe to operate on
input double MaxSpread_pips = 2.0; // max allowed spread in pips 
input int SessionStartHour = 7; // start trading hour (server time)
input int SessionEndHour = 12; // end trading hour (server time)
input double PartialCloseProfit_pips = 6.0; // profit threshold to trigger first partial close
input int PartialCloseSteps = 2; // how many partial close steps
input double PartialClosePercent = 0.5; // percent of bucket to close each partial step (0-1)
input ulong EA_Magic = 123456;

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
 int partial_steps_done;
};

Bucket buckets[];
int ATRHandle;

//---- global arrays for stats 
double volArray[];
double bodyRatioArray[];
int dirArray[];

double vol_ewma_arr[]; // smoothed vol

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
  buckets[i].partial_steps_done = 0;
 }
 
 ATRHandle = iATR(_Symbol, Timeframe, ATR_period);
 
 digits_adjust = (int) SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
 pip = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
 
 Print("Stateful_Scalper_EA initialized on ", _Symbol);
 return (INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+ 
void OnDeinit(const int reason) {}

//+------------------------------------------------------------------+
void OnTick() {
 // operate only once per new bar 
 static datetime lastTime = 0;
 datetime now = TimeCurrent();
 if ((now == lastTime) return;
 lastTime = now;
 
 MqlRates rates[];
 int bars = CopyRates(_Symbol, Timeframe, 0, LookbackStats + 10, rates);
 if (bars <= 0) return;
 
 // session filter - server time 
 MqlDateTime dt;
 TimeToStruct(now, dt);
 if (!InSession(dt.hour)) return;
 
 // quick filter: spread 
 double spread = (SymbolInfoDouble(_Symbol, SYMBOL_ASK) - SymbolInfoDouble(_Symbol, SYMBOL_BID)) / pip;
 if (spread > MaxSpread_pips) return;
 
 // compute feature arrays
 BuildFeatureArrays(rates, bars);
 
 // compute features for latest candle (index 0)
 double vol_z = VolumeZscore(0);
 double vol_smoothed = VolumeSmoothed(0);
 double body_ratio = BodyRatio(0);
 int direction = CandleDirection(0); // 1 long, -1 short, 0 neutral
 double wick_asym = WickAsymmetry(0);
 
 double atr;
 CopyBuffer(ATRHandle, 0, 0, 1, atr);
 
 // outlier detection
 double mdist = MahalanobisLikeDistance(0);
 
 double momentum = ComputeMomentumScore(direction, vol_z, body_ratio, wick_asym);
 
 // manage existing buckets & trades
 ManageBuckets(atr);
 
 // entry logic: open new bucket or add to existing 
 if (momentum >= MomentumThreshold && vol_z >= VolumeZMin && body_ratio >= BodyRatioMin && mdist > 1.5) { // try add to existing long bucket first 
  int idx = FindOpenBucket(1);
  if (idx >= 0 && buckets[idx].trades_count < MaxTradesPerBucket && momentum >= AddThreshold) { OpenAndRegisterTrade(1); } else { // open new long bucket
   CreateBucketAndTrade(1, "momentum_entry");
  }
 } else if (momentum <= -MomentumThreshold && vol_z <= -VolumeZMin && body_ratio >= BodyRatioMin && mdist > 1.5) {
  int idx = FindOpenBucket(-1);
  if (idx >= 0 && buckets[idx].trades_count < MaxTradesPerBucket && momentum <= -AddThreshold) OpenAndRegisterTrade(-1);
  else CreateBucketAndTrade(-1, "momentum_entry");
 }
 
 // optional: prune old buckets by time or equity 
 PruneBucketsByRules();
}

//+------------------------------------------------------------------+ 
void BuildFeatureArrays(const MqlRates &rates[], int bars) {
 ArrayResize(volArray, bars);
 ArrayResize(bodyRatioArray, bars);
 ArrayResize(dirArray, bars);
 ArrayResize(vol_ewma_arr, bars);
 
 // raw arrays 
 for (int i = 0; i < bars; i++) {
  double vol = (double) rates[i].tick_volume;
  volArray[i] = vol;
  double body = MathAbs(rates[i].close - rates[i].open);
  double total = rates[i].high - rates[i].low;
  bodyRatioArray[i] = total > 0 ? body / total : 0.0;
  dirArray[i] = (rates[i].close > rates[i].open) ? 1 : ((rates[i].close < rates[i].open) ? -1 : 0);
 }
 
 // EWMA smoothing for volume 
 double alpha = 2.0 / (VolEWMA_period + 1);
 int n = ArraySize(volArray) - 1;
 if (n <= 0) return;
 vol_ewma_arr[n] = volArray[n];
 for (int i = n - 1; i >= 0; i--) vol_ewma_arr[i] = alpha * volArray[i] + (1 - alpha) * vol_ewma_arr[i + 1];
 
 // simple Kalman-like one-dimensional smoother (process noise small) 
 // kalman gain adaptive isn't fully implemented; using a lightweight smoother 
 double estimate = vol_ewma_arr[n];
 double P = 1.0;
 double Q = 0.0001; // process noise double R = 1.0;    // measurement noise 
 for (int i = n - 1; i >= 0; i--) { // predict 
  P = P + Q; // update double 
  K = P / (P + R);
  estimate = estimate + K * (vol_ewma_arr[i] - estimate);
  P = (1 - K) * P;
  vol_ewma_arr[i] = estimate;
 }
}

//+------------------------------------------------------------------+ 
// Volume Z-score using LookbackStats (on index 'idx', 0 = latest) 
double VolumeZscore(int idx) {
 int n = MathMin(LookbackStats, ArraySize(volArray) - 1);
 if (n <= 2) return 0.0;
 
 double mean = 0.0, std = 0.0;
 for (int i = 1; i <= n; i++) mean += vol_ewma_arr[i];
 mean /= n;
 for (int i = 1; i <= n; i++) std += MathPow(vol_ewma_arr[i] - mean, 2);
 std = MathSqrt(std / (n - 1));
 double v = vol_ewma_arr[idx];
 if (std == 0.0) return 0.0;
 return (v - mean) / std;
}

//+------------------------------------------------------------------+ 
// Smoothed volume value 
double VolumeSmoothed(int idx) {
 if (idx < ArraySize(vol_ewma_arr)) return vol_ewma_arr[idx];
 return volArray[idx];
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
 MqlRates r[1];
 if (CopyRates(_Symbol, Timeframe, idx, 1, r) <= 0) return 0.0;
 
 double upper = r[0].high - MathMax(r[0].open, r[0].close);
 double lower = MathMin(r[0].open, r[0].close) - r[0].low;
 double range = r[0].high - r[0].low;
 if (range <= 0) return 0.0;
 
 return (upper - lower) / range; // positive => upper wick bigger
}

//+------------------------------------------------------------------+ 
// Mahalanobis-like distance using vol and bodyRatio (1D scaled approx) 
double MahalanobisLikeDistance(int idx) {
 int n = MathMin(LookbackStats, ArraySize(volArray) - 1);
 if (n <= 5) return 0.0;
 
 // compute means 
 double mean_v = 0, mean_b = 0;
 for (int i = 1; i <= n; i++) {
  mean_v += vol_ewma_arr[i];
  mean_b += bodyRatioArray[i];
 }
 mean_v /= n;
 mean_b /= n;
 
 // compute variances (diagonal approx) 
 double var_v = 0, var_b = 0;
 for (int i = 1; i <= n; i++) {
  var_v += MathPow(vol_ewma_arr[i] - mean_v, 2);
  var_b += MathPow(bodyRatioArray[i] - mean_b, 2);
 }
 var_v /= (n - 1);
 var_b /= (n - 1);
 if (var_v <= 0) var_v = 1.0;
 if (var_b <= 0) var_b = 0.01;
 
 double dv = vol_ewma_arr[idx] - mean_v;
 double db = bodyRatioArray[idx] - mean_b;
 double dist2 = dvdv / var_v + dbdb / var_b;
 
 return MathSqrt(dist2);
}

//+------------------------------------------------------------------+ 
// Composite momentum score (simple linear combination, tweak weights)
double ComputeMomentumScore(int dir, double vol_z, double body_ratio, double wick_asym) {
 double s = 0.0;
 s += dir * 1.2; // base direction weight 
 s += vol_z * 0.9; // volume abnormality 
 s += body_ratio * 1.5; // decisive body weight 
 s += (-MathAbs(wick_asym)) * 0.4; // penalize strong wick asymmetry 
 return s;
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
 for (int i = 0; i < ArraySize(buckets); i++)
  if (!buckets[i].active) { freeIdx = i; break; } if (freeIdx < 0) return; // no space
 
 // initialize bucket 
 buckets[freeIdx].active = true;
 buckets[freeIdx].direction = direction;
 buckets[freeIdx].trades_count = 0;
 buckets[freeIdx].batch_lot = 0.0;
 buckets[freeIdx].avg_entry = 0.0;
 buckets[freeIdx].reason = reason;
 buckets[freeIdx].partial_steps_done = 0;
 
 // open first trade for this bucket
 OpenAndRegisterTrade(direction);
 
 // set batch SL price based on settings (using ATR if available) 
 double price = (direction == 1) ? SymbolInfoDouble(_Symbol, SYMBOL_BID) : SymbolInfoDouble(_Symbol, SYMBOL_ASK);
 
 double atr;
 CopyBuffer(ATRHandle, 0, 0, 1, atr);
 
 if (atr <= 0) atr = BucketSL_pipspip;
 double slpips = MathMax(BucketSL_pips, atr / pip * 1.2);
 double sl = (direction == 1) ? price - slpipspip : price + slpips * pip;
 buckets[freeIdx].batch_sl_price = sl;
}

//+------------------------------------------------------------------+ 
void OpenAndRegisterTrade(int direction) {
 double price = (direction == 1) ? SymbolInfoDouble(_Symbol, SYMBOL_ASK) : SymbolInfoDouble(_Symbol, SYMBOL_BID);
 double estimated_sl_pips = BucketSL_pips;
 
 // if a bucket exists use its SL estimate 
 int idx = FindOpenBucket(direction);
 if (idx >= 0) {
  double avg = buckets[idx].avg_entry; 
  if (avg > 0) estimated_sl_pips = MathAbs((price - buckets[idx].batch_sl_price) / pip);
 }
 
 double lot = ComputeLotSizeDynamic(estimated_sl_pips);
 if (lot < BaseLot) lot = BaseLot;
 
 trade.SetExpertMagicNumber(EA_Magic);
 trade.SetDeviationInPoints(10);
 bool ok = false;
 if (direction == 1) ok = trade.Buy(lot, NULL, price, 0, 0, "bucket_entry");
 else ok = trade.Sell(lot, NULL, price, 0, 0, "bucket_entry");
 
 if (!ok) { Print("Trade open failed: ", trade.ResultRetcode(), " - ", trade.ResultRetcodeDescription()); return; }
 
 ulong t = trade.ResultOrder(); // register into bucket (create if missing) 
 int bidx = FindOpenBucket(direction);
 if (bidx < 0) { // find first inactive bucket and assign 
  for (int i = 0; i < ArraySize(buckets); i++)
   if (!buckets[i].active) { bidx = i; break; } if (bidx < 0) return;
  buckets[bidx].active = true;
  buckets[bidx].direction = direction;
  buckets[bidx].trades_count = 0;
  buckets[bidx].batch_lot = 0;
  buckets[bidx].avg_entry = 0; // set default SL 
  
  double atr;
  CopyBuffer(ATRHandle, 0, 0, 1, atr);
  
  double price_now = (direction == 1) ? SymbolInfoDouble(_Symbol, SYMBOL_BID) : SymbolInfoDouble(_Symbol, SYMBOL_ASK);
  double slpips = MathMax(BucketSL_pips, atr / pip * 1.2);
  buckets[bidx].batch_sl_price = (direction == 1) ? price_now - slpipspip : price_now + slpipspip;
 }
 
 TradeInfo ti;
 ti.ticket = t;
 ti.open_price = price;
 ti.lot = lot;
 ti.open_time = TimeCurrent();
 
 int pos = buckets[bidx].trades_count;
 if (pos < 32) {
  buckets[bidx].trades[pos] = ti;
  buckets[bidx].trades_count++;
  buckets[bidx].batch_lot += lot; // update avg entry 
  if (buckets[bidx].avg_entry == 0.0) buckets[bidx].avg_entry = price;
  else buckets[bidx].avg_entry = (buckets[bidx].avg_entry * (buckets[bidx].trades_count - 1) + price) / buckets[bidx].trades_count;
  
  // ensure batch SL exists: recalc
  RecalcBucketSL(bidx);
  
  PrintFormat("Opened trade %I64u dir=%d lot=%.2f price=%.5f", t, direction, lot, price);
  
 }
}

//+------------------------------------------------------------------+
// Dynamic lot sizing based on account balance and estimated SL pips 
// riskPercent = RiskPercentPerBucket (percentage of balance we're willing to lose if SL hit) 
// formula: lot = riskCurrency / (sl_pips * pip_value_per_lot)

double ComputeLotSizeDynamic(double sl_pips) {
 double balance = AccountInfoDouble(ACCOUNT_BALANCE);
 double riskCurrency = balance * (RiskPercentPerBucket / 100.0);
 if (sl_pips <= 0) sl_pips = BucketSL_pips;
 
 // get tick/currency conversion 
 double tick_value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
 double tick_size = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
 if (tick_value <= 0 || tick_size <= 0) { // fallback to approximate sizing using notional per pip for FX majors (assume 10 per pip per lot)
  double approx_perpip = 10.0;
  double lot = riskCurrency / (sl_pips * approx_perpip);
  return NormalizeLot(lot);
 }
 
 double pip_in_ticks = pip / tick_size; // number of ticks per pip 
 double value_per_pip_per_lot = tick_value * pip_in_ticks; // account currency per pip per lot
 if (value_per_pip_per_lot <= 0) value_per_pip_per_lot = 10.0; // safeguard
 
 double lot = riskCurrency / (sl_pips * value_per_pip_per_lot);
 
 return NormalizeLot(lot);
}

//+------------------------------------------------------------------+
double NormalizeLot(double lot) {
 double minlot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
 double maxlot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
 double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
 
 if (minlot <= 0 || maxlot <= 0 || step <= 0) return BaseLot;
 
 double rounded = MathFloor(lot / step + 0.0000001) * step;
 if (rounded < minlot) rounded = minlot;
 if (rounded > maxlot) rounded = maxlot;
 
 return NormalizeDouble(rounded, 2);
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
  double expected_atr = iATR(_Symbol, Timeframe, ATR_period, 1);
  if (expected_atr > 0 && (atr > expected_atr * 1.8 || atr < expected_atr * 0.5)) {
   // volatility changed drastically -> close bucket to avoid scalping in unstable regime
   CloseBucket(i, "volatility_regime_change");
   continue;
  }
  
  // partial close logic: if bucket profit reaches threshold, close percent
  double bucketPL = GetBucketProfit(i);
  double bucketPL_pips = ProfitToPips(bucketPL, buckets[i].direction);
  if (buckets[i].partial_steps_done < PartialCloseSteps && bucketPL_pips >= PartialCloseProfit_pips * (1 + buckets[i].partial_steps_done)) {
   // close percent of bucket
   double toCloseLots = buckets[i].batch_lot * PartialClosePercent;
   if (toCloseLots > 0) ClosePartialBucket(i, toCloseLots, "partial_profit");
   buckets[i].partial_steps_done++;
   // recalc avg entry and lot
  }
  
  // time-based or reason-based pruning could be added here
  
 }
}

//+------------------------------------------------------------------+ 
// Close a percentage (lots) of a bucket (round-robin over trades) 
void ClosePartialBucket(int idx, double lotsToClose, string reason) {
 if (idx < 0 || idx >= ArraySize(buckets)) return;
 double remaining = lotsToClose;
 for (int t = buckets[idx].trades_count - 1; t >= 0 && remaining > 0; t--) {
  ulong ticket = buckets[idx].trades[t].ticket;
  if (!PositionSelectByTicket(ticket)) continue;
  
  double closeLots = MathMin(PositionGetDouble(POSITION_VOLUME), remaining); // close partial by ticket using trade API (PositionClosePartial not universally available)
  bool closed = trade.PositionClose(ticket, closeLots);
  if (closed) {
   remaining -= closeLots;
   buckets[idx].batch_lot -= closeLots; // reduce trade record lot 
   buckets[idx].trades[t].lot -= closeLots;
   if (buckets[idx].trades[t].lot <= 0.000001) RemoveTradeFromBucket(idx, t); // remove trade from list (compact) 
  }
 }
 PrintFormat("Partial close bucket %d reason=%s closedLots=%.2f", idx, reason, lotsToClose - remaining);
}

//+------------------------------------------------------------------+ 
void RemoveTradeFromBucket(int bidx, int tindex) {
 for (int k = tindex; k < buckets[bidx].trades_count - 1; k++) buckets[bidx].trades[k] = buckets[bidx].trades[k + 1];
 buckets[bidx].trades_count--;
}

//+------------------------------------------------------------------+ 
void CloseBucket(int idx, string reason) {
 if (idx < 0 || idx >= ArraySize(buckets)) return;
 PrintFormat("Closing bucket %d reason=%s", idx, reason);
 for (int t = 0; t < buckets[idx].trades_count; t++) {
  ulong ticket = buckets[idx].trades[t].ticket;
  if (PositionSelectByTicket(ticket)) trade.PositionClose(ticket); // close full position
 } // reset bucket 
 buckets[idx].active = false;
 buckets[idx].trades_count = 0;
 buckets[idx].batch_lot = 0.0;
 buckets[idx].avg_entry = 0.0;
 buckets[idx].partial_steps_done = 0;
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
bool PositionSelectByTicket(ulong ticket) {
 for (int i = 0; i < PositionsTotal(); i++) {
  if (PositionGetTicket(i) == ticket) return true;
 }
 return false;
}

//+------------------------------------------------------------------+ 
// Compute bucket profit in account currency by summing positions' profit for trades in bucket

double GetBucketProfit(int idx) {
 if (idx < 0 || idx >= ArraySize(buckets)) return 0.0;
 double sum = 0.0;
 for (int i = 0; i < PositionsTotal(); i++) {
  if (!PositionSelectByIndex(i)) continue;
  ulong ticket = PositionGetInteger(POSITION_TICKET);
  double profit = PositionGetDouble(POSITION_PROFIT); // check if this ticket belongs to our bucket 
  for (int t = 0; t < buckets[idx].trades_count; t++)
   if (buckets[idx].trades[t].ticket == ticket) sum += profit;
 }
 return sum;
}
//+------------------------------------------------------------------+
// Convert profit in account currency to approximate pips for bucket direction
double ProfitToPips(double profit, int direction) { // estimate pips using value per pip per lot and total lots 
 double tick_value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
 double tick_size = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
 if (tick_value <= 0 || tick_size <= 0) return 0.0;
 double pip_in_ticks = pip / tick_size;
 double value_per_pip_per_lot = tick_value * pip_in_ticks;
 if (value_per_pip_per_lot <= 0) return 0.0;
 double lots = 0.0;
 for (int t = 0; t < buckets[idx].trades_count; t++) lots += buckets[idx].trades[t].lot;
 if (lots <= 0) return 0.0;
 double pips = profit / (value_per_pip_per_lot * lots);
 return pips;
}
//+------------------------------------------------------------------+ 
// Check if current hour is inside session
bool InSession(int hour) {
 if (SessionStartHour <= SessionEndHour) return (hour >= SessionStartHour && hour < SessionEndHour); // wrap-around (e.g., start 20, end 6)
 return (hour >= SessionStartHour || hour < SessionEndHour);
}