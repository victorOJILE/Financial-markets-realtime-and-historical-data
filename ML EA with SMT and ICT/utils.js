const tradeDenied = 'XAUUSD entry denied by volatile candle';

function IsBullish(candle) {
 return candle.close > candle.open;
}

// bool IsEngulfingCandle(MqlRates &prevCandle, MqlRates &candle, bool type) {
function IsEngulfingCandle(prevCandle, candle, isHigh) {
 if(!isHigh) { // Bullish
  return !IsBullish(prevCandle) && candle.close - candle.open > prevCandle.open - prevCandle.close && candle.close > prevCandle.open;
 } else {
  return IsBullish(prevCandle) && candle.open - candle.close > prevCandle.close - prevCandle.open && candle.close < prevCandle.open;
 }
}

// bool IsHammer(MqlRates &candle) {
function IsHammer(candle) {
 return ((Math.abs(candle.open - candle.close) / (candle.high - candle.low)) * 100) <= 40;
}

// bool IsFractal(MqlRates &prices[], int index, bool isHigh) {
function IsFractal(prices, index, isHigh) {
 let score = 0;
 
 if (isHigh) {
  let current = prices[index].high;
  if (prices[index - 1].high < current) {
   if (prices[index - 2].high < current) score++;
   score++;
  }
  if (score == 0) return false;
  
  if (prices[index + 1].high < current) {
   if (prices[index + 2].high < current) score++;
   score++;
  }
 } else {
  let current = prices[index].low;
  if (prices[index - 1].low > current) {
   if (prices[index - 2].low > current) score++;
   score++;
  }
  if (score == 0) return false;
  
  if (prices[index + 1].low > current) {
   if (prices[index + 2].low > current) score++;
   score++;
  }
 }
 
 return score == 4;
}

// Todo: Invalid GetOB logic
function GetOB(rates, swing) {
 /*let i = swing.end, end = swing.start;
 
 let OBCandle = {};
 
 if (swing.isBullish) {
  for (; i < end + 1; i--) {
   // Get order blocks
   if (!((!isBull(rates[i]) && IsHammer(rates[i])) || i == end)) continue; // Bearish hammer candle during bullish impulse or end of loop
   
   COrderBlock * newOB = new OrderBlocks();
   newOB.Type = OB_BUY;
   newOB.Index = i;
   newOB.fvg = NULL;
   
   if (i == end) {
    // if prev candle body is smaller than cur candle body 
    bool smallestBody = MathAbs(rates[i + 1].close - rates[i + 1].open) < MathAbs(rates[i].close - rates[i].open);
    OBCandle = rates[i + 1];
    newOB.LowerBound = MathMin(rates[i + 1].low, rates[i].low);
   } else {
    OBCandle = rates[i];
    newOB.LowerBound = OBCandle.low;
   }
   
   newOB.UpperBound = MathMax(OBCandle.close, OBCandle.open);
   newOB.CreatedTime = OBCandle.time;
   
   
   Add(newOB);
  }
 } else {
  for (; i < end + 1; i--) {
   // Get order blocks
   if (!((isBull(rates[i]) && IsHammer(rates[i])) || i == end)) continue; // Bullish hammer candle during bearish impulse or end of loop
   
   COrderBlock * newOB = new OrderBlocks();
   newOB.Type = OB_SELL;
   newOB.Index = i;
   newOB.fvg = NULL;
   
   if (i == end) {
    // if prev candle body is smaller than cur candle body
    bool smallestBody = MathAbs(rates[i + 1].close - rates[i + 1].open) < MathAbs(rates[i].close - rates[i].open);
    OBCandle = rates[i + 1];
    newOB.UpperBound = MathMax(rates[i + 1].high, rates[i].high);
   } else {
    OBCandle = rates[i];
    newOB.UpperBound = OBCandle.high;
   }
   
   newOB.LowerBound = MathMin(OBCandle.close, OBCandle.open);
   newOB.CreatedTime = OBCandle.time;
   
   Add(newOB);
  }
 }*/
}

// void GetFVG(MqlRates & rates[], int curIndex, CandleData & candle, CandleData & prevCandle, bool dir) {
function GetFVG(rates, curIndex, candle, prevCandle) {
 // Determine direction
 let dir = IsBullish(rates[2]);
 
 if (dir) { // Bullish
  if (IsFVG(rates, 1, dir)) {
   // Bullish FVG Check in upward move
   if (!IsFVG(rates, 2, dir)) {
    // Previous candle is not an fvg
    candle.fvgType = 0;
    candle.fvgStart = candle.low;
    candle.fvgEnd = rates[3].high;
   } else {
    // Since previous candle is also an fvg,
    // update the range
    if (prevCandle.fvgType == 0 && candle.low > prevCandle.fvgEnd) prevCandle.fvgStart = candle.low;
   }
  }
 } else {
  if (IsFVG(rates, 1, dir)) {
   // Bearish FVG Check in downward move
   if (!IsFVG(rates, 2, dir)) {
    // Previous candle is not an fvg
    candle.fvgType = 1;
    candle.fvgStart = candle.high;
    candle.fvgEnd = rates[3].low;
   } else {
    // Since previous candle is also an fvg,
    // update the range
    if (prevCandle.fvgType == 1 && candle.high < prevCandle.fvgEnd) prevCandle.fvgStart = candle.high;
   }
  }
 }
}

// bool IsFVG(Mqlrates & rates[], int index, bool dir) {
function IsFVG(rates, index, dir) {
 if (dir) {
  let hasGap = rates[index].low > rates[index + 2].high;
  let gap = rates[index].low - rates[index +2].high;
  let body = Math.abs(rates[index +1].open - rates[index +1].close);
  
  return hasGap && (gap / body) * 100 >= 50;
 } else {
  let hasGap = rates[index].high < rates[index + 2].low;
  let gap = rates[index +2].low - rates[index].high;
  let body = Math.abs(rates[index +1].open - rates[index +1].close);
  
  return hasGap && (gap / body) * 100 >= 50;
 }
}

function Slope(rates, isHigh) {
 let isBullish = rates[1].close > rates[1].open;
 let secIsBullish = rates[2].close > rates[2].open;
 
 if (isHigh) {
  if(IsEngulfingCandle(rates[2], rates[1], isHigh)) return true;
  if (!isBullish && !secIsBullish && rates[1].close < Math.min(rates[3].close, rates[3].open)) return true;
  
  return !isBullish && rates[1].close < Math.min(rates[3].close, rates[3].open);
 }
 
 if(IsEngulfingCandle(rates[2], rates[1])) return true;
 if (isBullish && secIsBullish && rates[1].close > Math.max(rates[3].close, rates[3].open)) return true;
 
 return isBullish && rates[1].close > Math.max(rates[3].close, rates[3].open);
}

// bool CandleSlopeBullish(Mqlrates y[]) {
function CandleSlopeBullish(y) {
 let n = y.length ;
 
 // Calculate slope of the regression line
 let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
 
 for (let i = 1; i < n; i++) {
  sumX += i;
  sumY += y[i];
  sumXY += i * y[i];
  sumX2 += i * i;
 }
 
 let slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
 
 return slope > 0.0;
}

// Todo: mql5 function convention
function GetTrueRanges(prices, lookback) {
 let ranges = [];
 
 for (let i = 1; i < lookback + 1; i++) {
  let current = prices[i];
  let a = Math.abs(current.low - prices[i + 1].close);
  let b = Math.abs(current.high - prices[i + 1].close);
  ranges.push(Math.max(a, b, current.high - current.low));
 }
 
 return ranges;
}

// void addEarlyIRL(Type & swings[], Mqlrates &prices[], bool isHigh) {
function addEarlyIRL(swings, prices, isHigh) {
 let found = 0;
 for (let i = 11; i < swings[0] - 1; i++) {
  if (IsFractal(prices, i, isHigh)) {
   found++;
   swings.splice(1, 0, i);
   if (found > 1) break;
  }
 }
}

// bool CandleNotVolatile(Mqlrates &prices[]) {
function CandleNotVolatile(prices) {
 let rangePerc = GetTrueRanges(prices, 10);
 let current = GetTrueRanges(prices, 1)[0];
 
 let volatile = current < Percentile(rangePerc, volatilityThresholdPercentile);
 if(!volatile) console.log(tradeDenied);
 
 return volatile;
}

// double Percentile(int &candleSizes, int percentile) {
function Percentile(candleSizes, percentile) {
 const sortedData = [...candleSizes].sort((a, b) => a - b);
 const n = sortedData.length;
 
 // Calculate the rank
 const rank = (percentile / 100) * (n - 1);
 
 // Handle integer rank (no interpolation needed)
 if (Number.isInteger(rank)) {
  return sortedData[rank];
 }
 
 // Handle fractional rank (interpolation for a more precise result)
 const lowerIndex = Math.floor(rank);
 const upperIndex = Math.ceil(rank);
 const lowerValue = sortedData[lowerIndex];
 const upperValue = sortedData[upperIndex];
 const weight = rank - lowerIndex;
 
 return (lowerValue + weight * (upperValue - lowerValue)) * 1.5;
}

function getCombinations(arr, size, callback) {
 function generate(c, start) {
  // If the current combination has the desired size, add it to the result
  if (c.length === size) {
   callback(c);
   return;
  }
  
  // Iterate through the array to build combinations
  for (let i = start; i < arr.length; i++) {
   c.push(arr[i]);
   generate(c, i + 1);
   c.pop(); // Backtrack
  }
 }
 generate([], 0);
}

