//+------------------------------------------------------------------+
//|                                     PriceRejectionDetector.mq5 |
//|                                  Copyright 2025, Your Company  |
//|                                       https://www.yourco.com   |
//+------------------------------------------------------------------+
#property copyright "2025, VictorDMN"
#property link "https://www.ojilevictor.com"
#property version "1.00"

// Enum for rejection types
enum RejectionType {
  SELL_SIDE_REJECTION = 1, // Sell side rejection (potential buy signal)
  BUY_SIDE_REJECTION = 2 // Buy side rejection (potential sell signal)
};

// --- Parameters for sensitivity (can be adjusted) ---
double minWickBodyRatio = 2.0; // Wick must be at least X times the size of the real body for hammer-like rejections
double minBodyRangeRatio = 0.2; // Real body should be small relative to the total candle range for hammer-like rejections
double minRejectionStrength = 0.6; // How much of the candle's range should be "rejected" for hammer-like rejections

// Helper function to determine if a candle is bullish
bool IsBullish(const MqlRates &candle) {
 return candle.close > candle.open;
}

void DetectPriceRejection(string symbol, ENUM_TIMEFRAMES timeframe, int numCandlesToCheck) {
 // Variable to store candles data
 MqlRates rates[];
 if (CopyRates(symbol, timeframe, 0, numCandlesToCheck + 1, rates) == -1) {
  // Failed to copy rates
  Print("Error copying rates: ", GetLastError());
  return;
 }
 
 MqlRates current = rates[1];
 MqlRates prev = rates[2];
 
 if (timeframe >= PERIOD_30) { // Only use hammers on higher timeframes
  double open = current.open;
  double close = current.close;
  double high = current.high;
  double low = current.low;
  
  double realBody = MathAbs(close - open);
  double candleRange = high - low;
  
  // Avoid division by zero
  if (candleRange == 0) candleRange = DBL_EPSILON; // Prevent division by zero for doji or very small candles
  
  double upperWick = high - MathMax(open, close);
  double lowerWick = MathMin(open, close) - low;
  
  // --- 1. General Price Rejection (Hammer-like / Pin bar concept) ---
  // Check for sell-side rejection (potential buy signal)
  if (!IsBullish(prev) && lowerWick / (realBody + DBL_EPSILON) >= minWickBodyRatio &&
   realBody / candleRange <= minBodyRangeRatio &&
   (MathMin(open, close) - low) / candleRange >= minRejectionStrength)
   return EnterTrade(SELL_SIDE_REJECTION, true);
  
  // Check for buy-side rejection (potential sell signal)
  if (IsBullish(prev) && upperWick / (realBody + DBL_EPSILON) >= minWickBodyRatio &&
   realBody / candleRange <= minBodyRangeRatio &&
   (high - MathMax(open, close)) / candleRange >= minRejectionStrength) return EnterTrade(BUY_SIDE_REJECTION, true);
 }
 // --- 2. Bullish Rejection: Candle sweeps previous low and closes >= body high ---
 if (!IsBullish(prev) && current.low < prev.low && current.close >= current.open && // Current candle is bullish or doji
  current.close >= MathMax(prev.open, prev.close) // Close above or near previous body high
 ) return EnterTrade(SELL_SIDE_REJECTION, true);
 
 // --- 3. Bullish Rejection: Bullish candle sweeps bodyLow of previous bullish candle, with 3rd candle being bearish ---
 MqlRates prevPrev = rates[3]; // 3rd candle back
 
 // Condition 1: Current candle is bullish
 // 2: Previous candle is bullish
 // 3: 3rd candle is bearish
 // 4: Current candle's low sweeps the body low of the previous (bullish) candle
 if (!IsBullish(prevPrev) && IsBullish(prev) &&
  current.low < MathMin(prev.open, prev.close) && // Current candle low sweeps previous bullish body low
  current.close > prev.close) return EnterTrade(SELL_SIDE_REJECTION, false);
 
 // --- 4. Bearish Rejection: Candle sweeps previous high and closes <= body low ---
 
 // Condition: Current candle high sweeps (goes above) the previous candle's high
 // And current candle closes below or at its own open (bearish or doji)
 // And current candle close is below the previous candle's body low (min(open, close))
 if (IsBullish(prev) && current.high > prev.high &&
  current.close <= current.open && // Current candle is bearish or doji
  current.close <= MathMin(prev.open, prev.close) // Close below or near previous body low
 ) return EnterTrade(BUY_SIDE_REJECTION, true);
 
 // --- 5. Bearish Rejection: Bearish candle sweeps bodyHigh of previous bearish candle, with 3rd candle being bullish ---
 
 // Condition 1: Current candle is bearish
 // Condition 2: Previous candle is bearish
 // Condition 3: Candle before previous (prevPrev) is bullish
 // Condition 4: Current candle's high sweeps the body high of the previous (bearish) candle
 if (IsBullish(prevPrev) && !IsBullish(prev) &&
  current.close < prev.close &&
  current.high > MathMax(prev.open, prev.close) // Current candle high sweeps previous bearish body high
 ) return EnterTrade(BUY_SIDE_REJECTION, false);
}

void OnTick() {
 // Only process on new bar close to avoid re-evaluating on every tick
 static datetime lastBarTime = 0;
 datetime currentBarTime = iTime(_Symbol, _Period, 0);
 
 if (currentBarTime != lastBarTime) {
  lastBarTime = currentBarTime;
  
  int candlesToCheck = 3; // You can make this an input parameter
  // Not enough bars available for full analysis
  if (Bars(_Symbol, timeframe) < candlesToCheck + 1) return;
  
  DetectPriceRejection(_Symbol, PERIOD_CURRENT, candlesToCheck);
 }
}
//+------------------------------------------------------------------+
void EnterTrade(RejectionType type, bool reversal) {
 if(reversal) {
  double volume[3];
  if(CopyVolume(symbol, PERIOD_CURRENT, 0, 3, volume) < 3 || volume[1] < volume[2]) return;
 }
 
 // Trade logic goes here
}