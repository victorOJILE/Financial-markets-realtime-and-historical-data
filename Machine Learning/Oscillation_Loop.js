// Struct to hold trend/range zone
struct Zone {
 int start_index;
 int end_index;
 string type; // "bullish", "bearish", "range"
};

Zone zones[];

// Check if candle is bullish
bool IsBullish(MqlRates rate) {
 return rate.close > rate.open;
}

// Check if candle is a ranging candle
bool IsRangingCandle(MqlRates rate) {
 return (MathAbs(rate.close - rate.open) < 0.3 * rate.high - rate.low);
}

// Check if volume is reduced compared to previous
bool IsVolumeReducing(int index) {
 if (index + 1 >= Bars) return true; // Avoid out of bounds
 return Volume[index] < Volume[index + 1];
}

// Detect trend/range zones with flexible logic
void DetectCandlePatternWithThreshold() {
 ArrayResize(zones, 0); // Clear old zones
 int right = 1; // Start from last closed candle
 int min_window = 5;
 int max_candles = Bars - 10;
 
 while (right + min_window < max_candles) {
  for (int window = min_window; right + window < max_candles; window++) {
   int left = right + window;
   
   int bull_count = 0, bear_count = 0, range_count = 0;
   int weak_opposite = 0;
   
   for (int i = right; i <= left; i++) {
    bool IsRanging = IsRangingCandle(i);
    
    if (IsBullish(i)) bull_count++;
    else bear_count++;
    
    // Check for weak opposite candle
    if(IsRanging && IsVolumeReducing(i)) weak_opposite++;
    
    if (IsRanging) range_count++;
   }
   
   int total = left - right + 1;
   string detected = "";
   
   // Bullish Trend with threshold allowance
   if (bull_count + weak_opposite >= total * 0.9) detected = "bullish";
   
   // Bearish Trend with threshold allowance
   else if (bear_count + weak_opposite >= total * 0.9) detected = "bearish";
   
   // Ranging zone (mostly small bodies)
   else if (range_count >= total * 0.7) detected = "range";
   
   if (detected != "") {
    // Right-expansion (oscillation)
    int r = right - 1;
    while (r >= 0) {
     bool match = false;
     
     if (detected == "bullish" && (IsBullish(r) || (IsRangingCandle(r) && IsVolumeReducing(r)))) match = true;
     else if (detected == "bearish" && (!IsBullish(r) || (IsRangingCandle(r) && IsVolumeReducing(r)))) match = true;
     else if (detected == "range" && IsRangingCandle(r)) match = true;
     
     if (!match) break;
     right = r;
     r--;
    }
    
    // Record zone
    int zone_index = ArraySize(zones);
    ArrayResize(zones, zone_index + 1);
    zones[zone_index].start_index = left;
    zones[zone_index].end_index = right;
    zones[zone_index].type = detected;
    
    // Shift to left
    right = left + 1;
    break;
   }
  }
  
  right++; // Keep scanning if nothing found
 }
}








let leftIndex = timeSeries.length - 5; // starting index
let rightIndex = timeSeries.length - 1; // starting index
let windowSize = 5; // initial window size
let direction = -1; // direction of expansion (left or right)

while (leftIndex >= 0) {
 // check the condition for the current window
 const window = timeSeries.slice(leftIndex, rightIndex + 1);
 const conditionMet = checkCondition(window);
 
 // if the condition is met, expand the window to the right else left
 direction = conditionMet ? 1 : -1;
 // if(!conditionMet) direction == 1 ? -1 : 1;
 
 // update the window indices
 if (direction === -1) {
  leftIndex--;
 } else {
  rightIndex = Math.min(rightIndex + 1, timeSeries.length - 1);
 }
 
 // check if we need to switch direction
 if (direction === 1 && !checkCondition(timeSeries.slice(leftIndex, rightIndex + 1))) direction = -1;
 
 // increment the window size if expanding to the left
 if (direction === -1) windowSize++;
}

// function to check the condition
function checkCondition(window) {
 // implement your condition checking logic here
 // return true if the condition is met, false otherwise
}