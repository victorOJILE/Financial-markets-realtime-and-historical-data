let tUsdHighD = 0; // Tracking xauusd high divergence 
let tUsdLowD = Infinity;
let tEurHighD = 0;
let tEurLowD = Infinity;

function CheckSMT(xauusd, xaueur) {
 let [swingHighsUsd, swingLowsUsd] = GetSwings(xauusd);
 let [swingHighsEur, swingLowsEur] = GetSwings(xaueur);
 
 let highs = Array.from(new Set(swingHighsUsd.concat(swingHighsEur)));
 highs.sort((a, b) => a - b);
 
 let lows = Array.from(new Set(swingLowsUsd.concat(swingLowsEur)));
 lows.sort((a, b) => a - b);
 
 let BuySignal = DetectSMT_FractalDivergence(lows);
 let SellSignal = DetectSMT_FractalDivergence(highs, 1);
 
 let confirmed = false;
 let signal = 0;
 
 if (BuySignal && SellSignal) {
  InitTrackers();
 } else {
  if (BuySignal) {
   signal = BuySignal;
   if(Slope(xauusd)) {
    InitTrackers();
    if (CandleNotVolatile(xauusd)) confirmed = true;
   }
  }
  
  if (SellSignal) {
   signal = SellSignal;
   if(Slope(xauusd, 1)) {
    InitTrackers(1);
    if (CandleNotVolatile(xauusd)) confirmed = true;
   }
  }
 }
 
 /*
 // Todo percentile should not check only xauusd
 // Check candle percentile
 
 //if(currentPerc > rangeRes) {Current candle percentile out of range. 
 console.log('Current candle: ', currentPerc, 'Range: ', rangeRes, 'Returning...');
 // Todo: Zero swing out
 
  return;
 }
 */
 
 return { confirmed, signal };
}

function DetectSMT_FractalDivergence(swings, isHigh) {
 let usdHigh = Math.max(tUsdHighD, xauusd[1].high);
 let usdLow = Math.min(tUsdLowD, xauusd[1].low);
 let usdClose = xauusd[1].close;
 let eurHigh = Math.max(tEurHighD, xaueur[1].high);
 let eurLow = Math.min(tEurLowD, xaueur[1].low);
 let eurClose = xaueur[1].close;
 
 
 for (let i = 0; i < swings.length; i++) {
  if (isHigh) {
   let h1 = xauusd[swings[i]].high;
   let h2 = xaueur[swings[i]].high;
   // || ((usdClose > h1 && eurClose <= h2) || (usdClose < h1 && eurClose >= h2)) 
   
   if ((usdHigh > h1 && eurHigh <= h2) || (usdHigh < h1 && eurHigh >= h2)) {
    let sweep = usdHigh > h1 ? 'XAUUSD' : 'XAUEUR';
    let pair = map[sweep];
    let recentHigh = sweep == 'XAUUSD' ? usdHigh : eurHigh;
    let valid = true;
    
    for (let j = swings[i]; j > 1; j--) {
     if (pair[j].high > recentHigh) valid = false;
    }
    // Todo: we need to gather divergence data
    // Body candle or wick divergence
    // A wick only divergence might consider third candle confirming past half of first candle,
    // check in Slope
    if (valid) {
     tUsdHighD = usdHigh;
     tEurHighD = eurHigh;
     
     return 2;
    }
   }
  } else {
   let l1 = xauusd[swings[i]].low;
   let l2 = xaueur[swings[i]].low;
   // Todo:
   // Also use differences in close price for detecting divergence (first condition)
   if ((usdLow > l1 && eurLow <= l2) || (usdLow < l1 && eurLow >= l2)) {
    let sweep = (usdLow < l1 && eurLow >= l2) ? 'XAUUSD' : 'XAUEUR';
    let pair = map[sweep];
    let recentLow = sweep == 'XAUUSD' ? usdLow : eurLow;
    let valid = true;
    
    for (let j = swings[i]; j > 1; j--) {
     if (pair[j].low < recentLow) valid = false;
    }
    
    if (valid) {
     tUsdLowD = usdLow;
     tEurLowD = eurLow;
     
     return 1;
    }
   }
  }
 }
 
 return false;
}

function GetSwings(prices) {
 let prevH = prices[1].high;
 let prevL = prices[1].low;
 let swingCountH = 0,
  swingCountL = 0;
 let swingHighs = [],
  swingLows = [];
 
 for (let i = 2; i < prices.length - 3 && (swingCountH < MaxSwings && swingCountL < MaxSwings); i++) {
  if(i < 10) {
   prevH = Math.max(prevH, prices[i].high);
   prevL = Math.min(prevL, prices[i].low);
   continue;
  }
  
  let curH = prices[i].high;
  let curL = prices[i].low;
  
  if (swingCountH < MaxSwings && curH > prevH && IsFractal(prices, i, 1)) {
   swingHighs.push(i);
   swingCountH++;
   prevH = curH;
  }
  
  if (swingCountL < MaxSwings && curL < prevL && IsFractal(prices, i)) {
   swingLows.push(i);
   swingCountL++;
   prevL = curL;
  }
 }
 
 // Let's add any fractal in between current candle and first swing
 if (swingHighs.length) addEarlyIRL(swingHighs, prices, true);
 if (swingLows.length) addEarlyIRL(swingLows, prices);
 
 return [swingHighs, swingLows];
}

function InitTrackers() {
 tUsdHighD = 0;
 tEurHighD = 0;
 tUsdLowD = Infinity;
 tEurLowD = Infinity;
}
