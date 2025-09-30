// Variables to track the Break of Structure setup
let isBullish = false; // bool
let isBearish = false; // bool
let rejectionCandleOpen = 0; // double
let isBOSConfirmed = false; // bool

// Function to check if a candle is bullish
// bool IsBullishCandle(int index) {
function IsBullishCandle(index) {
 return iClose(Symbol(), Period(), index) > iOpen(Symbol(), Period(), index);
}

// Function to check for Break of Structure (BOS)
function CheckBOS() { // bool
 // Step 1: Determine direction (bullish or bearish)
 if (!isBullish && !isBearish) {
  let highestHigh = 0; // double
  let lowestLow = 0; // double
  for (let i = 1; i < 20; i++) {
   if (iHigh(Symbol(), Period(), i) > highestHigh) {
    highestHigh = iHigh(Symbol(), Period(), i);
   }
   if (iLow(Symbol(), Period(), i) < lowestLow) {
    lowestLow = iLow(Symbol(), Period(), i);
   }
  }
  if (iClose(Symbol(), Period(), 0) > highestHigh) {
   isBullish = true;
  } else if (iClose(Symbol(), Period(), 0) < lowestLow) {
   isBearish = true;
  }
  return false;
 }
 
 // Step 2: Check for high/low break
 if (isBullish) {
  let highestHigh = 0; // double
  let consolidationCandles = 0; // int
  for (let i = 1; i < 20; i++) {
   if (iHigh(Symbol(), Period(), i) > highestHigh) {
    highestHigh = iHigh(Symbol(), Period(), i);
    consolidationCandles = 0;
   } else {
    consolidationCandles++;
    if (consolidationCandles >= 3 && iClose(Symbol(), Period(), 0) > highestHigh) { // iClose must be a bullish candle
     // Loop backwards to find the first bearish candle (rejection candle)
     for (let j = 1; j < 10; j++) {
      if (!IsBullishCandle(j)) {
       rejectionCandleOpen = iOpen(Symbol(), Period(), j);
       isBOSConfirmed = true;
       break;
      }
     }
    }
   }
  }
 }
 
 if (isBearish) {
  let lowestLow = 0; // double
  let consolidationCandles = 0; // int
  for (let i = 1; i < 20; i++) {
   if (iLow(Symbol(), Period(), i) < lowestLow) {
    lowestLow = iLow(Symbol(), Period(), i);
    consolidationCandles = 0;
   } else {
    consolidationCandles++;
    if (consolidationCandles >= 3 && iClose(Symbol(), Period(), 0) < lowestLow) { // iClose must be a bearish candle
     // Loop backwards to find the first bullish candle (rejection candle)
     for (let j = 1; j < 10; j++) {
      if (IsBullishCandle(j)) {
       rejectionCandleOpen = iOpen(Symbol(), Period(), j);
       isBOSConfirmed = true;
       break;
      }
     }
    }
   }
  }
 }
 
 return isBOSConfirmed;
}

// OnTick function
function OnTick() { // void
 // Check for Break of Structure (BOS)
 if (CheckBOS()) {
  // Check if current candle price has reached the rejection candle open
  if (isBullish && iLow(Symbol(), Period(), 0) <= rejectionCandleOpen) {
   // Open a long trade
  } else if (isBearish && iHigh(Symbol(), Period(), 0) >= rejectionCandleOpen) {
   // Open a short trade
  }
 }
}