Array.prototype.last = function() {
 return this[this.length - 1];
};

class CustomTrendFinder {
 constructor(params) {
  this.minR_squared = params.minR_squared;
  this.minPoints = params.minPoints;
  this.numTrees = params.numTrees;
  this.residualCheckPoints = params.residualCheckPoints;
  this.minResidualCheckPct = params.minResidualCheckPct;
 }
 
 findTrend(swingPoints) {
  if (swingPoints.length < this.minPoints) return null;
  
  const startTime = performance.now();
  let finder = this;
  // Use a setTimeout to allow the UI to update with the "Running" message
  setTimeout(() => {
   let allValidCandidates = [];
   try {
    console.log('Running algorithm...');
    const numbers = swingPoints.map((a, i) => i);
    let a = [];
    
    function callB(selectedPoints) {
     selectedPoints = selectedPoints.map(idx => swingPoints[idx]);
     a.push(selectedPoints.map(e => e.index).join(' '));
     // Perform linear regression
     const regressionResult = finder._calculateLinearRegression(selectedPoints);
     
     if (regressionResult && regressionResult.r_squared >= finder.minR_squared) {
      // 2. Backward Loop Residual Check
      const earlierPoints = swingPoints.filter(p => p.index > selectedPoints[0].index);
      //const internalPoints = swingPoints.filter(p => p.index )
      let valid = true;
      
      if (earlierPoints.length > 0) {
       // const isUptrend = regressionResult.slope > 0;
       
       for (let p of earlierPoints) {
        const predictedY = regressionResult.slope * p.index + regressionResult.intercept;
        const residual = p.price - predictedY;
        
        // if (!((isUptrend && residual > 0) || (!isUptrend && residual < 0))) {
        if (residual > 0) {
         valid = false;
         break;
        }
       }
      }
      
      if (valid) {
       regressionResult.points = selectedPoints;
       allValidCandidates.push(regressionResult);
      }
     }
    }
    
    // Generate combinations for sizes 3 to 7
    for (let size = 3; size <= numbers.length; size++) getCombinations(numbers, size, callB);
    
    // console.log(allValidCandidates.map(e => e.points.map(a => a.index).join(' ')));
    console.log(a);
    // 3. Ensemble and Voting
    if (allValidCandidates.length === 0) {
     console.log(`Algorithm finished. ❌ No valid trend line found.`);
     return null;
    }
    
    const foundTrend = allValidCandidates.reduce((best, current) => current.mse < best.mse ? current : best);
    
    const endTime = performance.now();
    const elapsedTime = (endTime - startTime).toFixed(2);
    
    if (foundTrend) {
     console.log(`Algorithm finished in ${elapsedTime}ms. ✅ Trend found and validated!`);
     console.log('R_Squared: ' + foundTrend.mse);
     console.log('Points used: ' + foundTrend.points.map(p => `(${p.index}, ${p.price.toFixed(2)})`).join(', '));
    } else {
     console.log(`Algorithm finished in ${elapsedTime}ms. ❌ No valid trend line found.`);
    }
   } catch (e) {
    console.error(e);
   }
  }, 10);
 }
 
 _calculateLinearRegression(points) {
  const n = points.length;
  if (n < 2) return null;
  
  let sum_x = 0,
   sum_y = 0,
   sum_xy = 0,
   sum_x2 = 0;
  points.forEach(p => {
   sum_x += p.index;
   sum_y += p.price;
   sum_xy += p.index * p.price;
   sum_x2 += p.index * p.index;
  });
  
  const denominator = n * sum_x2 - sum_x * sum_x;
  if (Math.abs(denominator) < 1e-6) return {
   r_squared: 0,
   slope: 0,
   intercept: sum_y / n
  }; // Horizontal line
  
  const slope = (n * sum_xy - sum_x * sum_y) / denominator;
  const intercept = (sum_y - slope * sum_x) / n;
  
  const y_mean = sum_y / n;
  let ss_total = 0,
   ss_residual = 0;
  points.forEach(p => {
   const predicted_y = slope * p.index + intercept;
   ss_total += Math.pow(p.price - y_mean, 2);
   ss_residual += Math.pow(p.price - predicted_y, 2);
  });
  
  const r_squared = (Math.abs(ss_total) < 1e-6) ? 1 : (1 - (ss_residual / ss_total));
  const mse = ss_residual / n;
  
  return { r_squared, slope, intercept, mse };
 }
}

/*for(let p of priceArray) for(let k in p) p[k] = Number(p[k]);


let loopCount = 150;
let len = priceArray.length;

function isBull(rates, i) {
 let ret = {
  bull: rates[i].close > rates[i].open,
  perc: 0
 };
 if(ret.bull) {
  if(rates[i].close > rates[i+1].high) ret.perc += 50;
  if(rates[i].close > rates[i+1].close &&  rates[i].close > rates[i+1].open) ret.perc += 25;
  if(Math.abs(rates[i].open - rates[i].high) > Math.abs(rates[i].open - rates[i].low)) ret.perc += 25;
 } else {
  if(rates[i].close < rates[i+1].low) ret.perc += 50;
  if(rates[i].close < rates[i+1].close && rates[i].close < rates[i+1].open) ret.perc += 25;
  if(Math.abs(rates[i].open - rates[i].high) < Math.abs(rates[i].open - rates[i].low)) ret.perc += 25;
 }
 
 return ret;
}

let dir1 = isBull(priceArray, 0).bull ? 1 : 0;
priceA[0].dir = dir1;
for(let i = 0; i < len -1; i++) {
 let dir = isBull(priceA, i);
 
 if(dir1 && !dir.bull && dir.perc < 50) priceA[i].dir = 1;
 else if(!dir1 && dir.bull && dir.perc < 50) priceA[i].dir = 0;
 else priceA[i].dir = dir.bull ? 1 : 0;
 
 dir1 = priceA[i].dir;
}
*/

function getCombinations(arr, size, callback) {
 function generate(currentCombination, start) {
  // If the current combination has the desired size, add it to the result
  if (currentCombination.length === size) {
   callback(currentCombination);
   return;
  }
  
  // Iterate through the array to build combinations
  for (let i = start; i < arr.length; i++) {
   currentCombination.push(arr[i]);
   generate(currentCombination, i + 1);
   currentCombination.pop(); // Backtrack
  }
 }
 generate([], 0);
}

function run(e, timer) {
 if (timer) priceA.push(e);
 else priceA = xauusd.slice();
 /*
 let data = [/*{
  start: priceA.length - 4,
  end: priceA.length - 4,
  isBullish: false
 }*];
 
 for (let i = priceA.length - 2; i >= 0; i--) {
  data.push(isBull(priceA, i));
 }
// console.log(data);
 let rates = priceA.slice().reverse();
 
 let i = 0, highs = [0], lows = [0];
 
 for (; i < rates.length -1;  i++) {
  if(rates[i].high > rates[highs.last()].high) {
   if(i - highs.last() <= 1) highs[highs.length -1] = i;
   else highs.push(i);
  }
  if(rates[i].low < rates[lows.last()].low) {
   if(i - lows.last() <= 1) lows[lows.length -1] = i;
   else lows.push(i);
  }
 }
 
 let OBCandle;
 i = 0;
 while (i < highs.length) {
  let newOB = {}, cur = highs[i];
  newOB.Type = "OB_SELL";
  
  // if prev candle body is smaller than cur candle body 
  let smallestBody = Math.abs(rates[cur + 1].close - rates[cur +1].open) < Math.abs(rates[cur].close - rates[cur].open);
  
  newOB.Index = smallestBody ? cur +1 : cur;
  OBCandle = rates[newOB.Index];
  newOB.LowerBound = Math.max(Math.min(rates[cur +1].open, rates[cur +1].close), Math.min(rates[cur].open, rates[cur].close));
 
  newOB.UpperBound = Math.max(OBCandle.high, rates[cur].high);
  newOB.CreatedTime = OBCandle.time;
   
  /*
   if (UseFVG) {
    CFVG* FVG = new CFVG();
    FVG.UpperBound = 0;
    FVG.LowerBound = 0;
    FVG.CreatedTime = TimeCurrent();
    FVG.Type = OB_BUY;
    newOB->fvg = FVG;
   }
  *
  highs[i++] = newOB;
 }
 i = 0;
 while (i < lows.length) {
  let newOB = {}, cur = lows[i];
  newOB.Type = "OB_BUY";
  
  // if prev candle body is smaller than cur candle body 
  let smallestBody = Math.abs(rates[cur +1].close - rates[cur + 1].open) < Math.abs(rates[cur].close - rates[cur].open);
  
  newOB.Index = smallestBody ? cur +1 : cur;
  OBCandle = rates[newOB.Index];
  newOB.LowerBound = Math.min(OBCandle.low, rates[cur].low);
 
  newOB.UpperBound = Math.min(Math.max(rates[cur +1].open, rates[cur +1].close), Math.max(rates[cur].open, rates[cur].close));
  newOB.CreatedTime = OBCandle.time;
   
  /*
   if (UseFVG) {
    CFVG* FVG = new CFVG();
    FVG.UpperBound = 0;
    FVG.LowerBound = 0;
    FVG.CreatedTime = TimeCurrent();
    FVG.Type = OB_BUY;
    newOB->fvg = FVG;
   }
  *
  
  lows[i++] = newOB;
 }
 */
 console.log(priceA);
 const RSquaredThreshold = 0.99; // User-defined threshold
 
 let prev = Infinity;
 let m_swing_points = [];
 
 for (let i = priceA.length - 3; i >= 2; i--) {
  let cur = priceA[i].low;
  if (cur < prev && IsFractal(priceA, i)) {
   m_swing_points.push({ price: cur, index: i });
   prev = cur;
  }
 }
 /*
 let swingLen = m_swing_points.length;
 if(swingLen > 3) {
  let best = {
   r_squared: -1,
   clusters: []
  };
  
  let allIdx = [];
  
  for(let k = 2; k < swingLen; k++) {
   // Work array to filter indexes
   let idx = m_swing_points.slice(0, k);

   while (idx.length > 4) {
    // Build sub-prices from current idx[]
    let subPrices = idx.map(e => e.price );
    let r = CalculateTrendLine(subPrices);
    let newIdx = [], newRes = [];
    for (let j = 0; j < idx.length; j++) {
     // Uptrend → keep BELOW line (negative residuals)
     // Downtrend → keep ABOVE line (positive residuals)
     let residual = LineMatch(r, j, subPrices[j], r.slope < 0);
     newRes.push(residual);
     if(residual < 0) {
      newIdx.push(idx[j]);
     }
    }
    console.log(newRes);
    // Store if better
    if (r.r_squared > best.r_squared) {
     best = r;
     best.clusters = newIdx;
     allIdx.push(newIdx);
    }
    
    // Update working set
    idx = newIdx;
   }
  }
  
  console.log(best);
  console.log(allIdx.map(e => e.map(a => a.index)));
  console.log(m_swing_points);
 }
 */
 /*
 // Check if there are enough swing points to analyze
 if (swingLen > 2) {
  let bestRSquared = -1.0;
  let bestPointsCount = 0;
  let bestPoints, currentPoints;
  
  // Iterate through all possible starting points
  for (let i = 0; i < swingLen - 2; i++) {
   // Start with the first three points
   currentPoints = m_swing_points.slice(i, i +3);
   let trendlineData = CalculateTrendLine(currentPoints);
   
   if(i > 0 && trendlineData.r_squared > bestRSquared && CheckForwardResiduals(currentPoints, m_swing_points, i -1, trendlineData, 0)) {
    bestRSquared = trendlineData.r_squared;
    bestPoints = currentPoints;
   }
  }
   for (let j = 100; j < swingLen -1; j++) {
    currentPoints = m_swing_points.slice(i, i +3);
    
    let currentRSquared = -1.0; // R-squared is 1.0 for two points
    
    // Progressively add more points and check the R-squared
    for (let k = j + 3; k < m_swing_points.length; k++) {
     // Add the new point to the temporary array
     currentPoints.push(m_swing_points[k]);
     
     let trendlineData = CalculateTrendLine(currentPoints);
     if (trendlineData.r_squared < currentRSquared) {
      // Remove the last point that broke the threshold
      currentPoints.pop();
     } else {
      currentRSquared = trendlineData.r_squared;
     }
    }
    
    // Check if this group is better than the current best
    let currentPointsCount = currentPoints.length;
    if (currentPointsCount > bestPoints.length && currentRSquared > RSquaredThreshold) {
     console.log(currentPoints, currentRSquared);
     bestRSquared = currentRSquared;
     bestPoints = currentPoints;
    }
   }
  
  //------------------------------------------------------------------+
  // REVERSED CHECK - Step 1: Initialize temporary array with the best points found
  //------------------------------------------------------------------+
 // let finalPoints = m_swing_points.slice(bestStartIndex, bestPointsCount + bestStartIndex);
  
  // Get the current best R-squared and point count as a baseline
  let finalRSquared = bestRSquared;
  let finalPointsCount = bestPointsCount;
  
  //------------------------------------------------------------------+
  // Step 2: Loop backward from the start of the current best group
  //------------------------------------------------------------------+
  /*for (let i = bestStartIndex - 1; i >= 0; i--) {
   // Prepend the new, earlier point to the array
   let newSize = finalPoints.length + 1;
   
   finalPoints = [m_swing_points[i]].concat(finalPoints.slice(1, newSize));
   
   // Recalculate R-squared for the extended group
   let newRSquared = CalculateRSquared(finalPoints);
   
   // Check if the R-squared is still within the threshold
   if (newRSquared >= RSquaredThreshold) {
    // Update the final best values if the extended group is valid
    finalRSquared = newRSquared;
    finalPointsCount = finalPoints.length;
   } else {
    // If the threshold is broken, remove the last added point and break the loop
    finalPoints.pop();
    break;
   }
  }
  *
  // Print the final result
  //console.log("  - R-Squared Value: ", finalRSquared);
  console.log("Best Trendline: ");
  console.table(bestPoints);
  console.log(bestRSquared);
  /*console.log("  - Start Point Time: ", finalPoints[0].time);
  console.log("  - End Point Time: ", finalPoints[finalPoints.length - 1].time);
  *
 }
 //console.log(priceA);
 console.table(m_swing_points);
 */
 /*
 let trendline = new CustomTrendFinder({
  minR_squared: RSquaredThreshold,
  minPoints: 3,
  numTrees: 100,
  residualCheckPoints: 10,
  minResidualCheckPct: 0.8
 });
 trendline.findTrend(m_swing_points);
 */
 
 candlesArr = priceA;
}
run();

function CheckForwardResiduals(points, swings, start, line, dir) {
 while (start > -1) {
  if (!LineMatch(line, swings[start].index, swings[start].price, dir)) return false;
  start--;
 }
 
 return true;
}

function LineMatch(line, x, y, dir) {
 return y - (line.slope * x + line.intercept);
 
 if (!dir && residual < 0) return false;
 if (dir && residual > 0) return false;
 
 return true;
}

function CalculateTrendLine(points) {
 let n = points.length;
 // Arrays for linear regression calculation
 let x = [],
  y = [];
 
 // Populate x and y arrays from the input array
 for (let i = 0; i < n; i++) {
  x[i] = i; // Use index as the independent variable (x)
  y[i] = points[i]; // Use price as the dependent variable (y)
 }
 
 // Calculate slope and intercept of the regression line
 let sumX = 0,
  sumY = 0,
  sumXY = 0,
  sumX2 = 0;
 
 for (let i = 0; i < n; i++) {
  sumX += x[i];
  sumY += y[i];
  sumXY += x[i] * y[i];
  sumX2 += x[i] * x[i];
 }
 
 let slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
 let intercept = (sumY - slope * sumX) / n;
 
 // Calculate R-squared
 let SS_total = 0,
  SS_residual = 0;
 let avgY = sumY / n;
 
 for (let i = 0; i < n; i++) {
  // Total sum of squares
  SS_total += Math.pow(y[i] - avgY, 2);
  
  // Residual sum of squares (error)
  let predictedY = slope * x[i] + intercept;
  SS_residual += Math.pow(y[i] - predictedY, 2);
 }
 
 // Handle division by zero
 if (SS_total == 0) return 1.0;
 
 let r_squared = 1 - (SS_residual / SS_total);
 let mse = SS_residual / n;
 
 return { slope, intercept, r_squared, mse };
}

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
 
 return score >= 3;
}
/*
let interval = setInterval(() => {
 run(priceArray[loopCount], 1);
 loopCount++;
}, 1000);
*/

//console.log(str);
/**
 * Returns the mean for a given dataset
 * @param {Array} dataset
 * @returns {number}
 */
Math.mean = function mean(dataset) {
 if (dataset.length == 1) return dataset[0];
 return dataset.reduce(function(a, b) {
  if (typeof b != 'number') throw new Error('Dataset array provided to Math.mean should contain only numbers!');
  return a + b;
 }) / dataset.length;
}

/**
 * Returns the median for a given dataset
 * @param {Array} dataset
 * @returns {number}
 */
Math.median = function median(dataset) {
 let len = dataset.length;
 if (len % 2 == 0) return (dataset[(len / 2) - 1] + dataset[(len / 2)]) / 2;
 
 return dataset[(len - 1) / 2];
}

/**
 * Returns the mode for a given dataset
 * @param {Array} dataset
 * @returns {number}
 */
Math.mode = function mode(dataset) {
 let count = dataset.reduce((a, b) => {
  if (a[b]) a[b]++;
  else a[b] = 1;
  
  return a;
 }, {});
 
 let mostOccur = dataset[0];
 for (let key in count)
  if (count[key] > count[mostOccur]) mostOccur = key;
 
 return mostOccur;
}

/**
 * Returns the variance for a given dataset
 * @param {number|Array} dataset
 * @param {number} dataset A number that represents the standard deviation of the dataset
 * @returns {number}
 */
Math.variance = function variance(dataset) {
 if (typeof dataset === "number") return dataset * dataset; // We assume that number is the standard deviation
 // else if dataset is an array
 else if (dataset.constructor.name !== "Array") throw new Error('Type of parameter provided to Math.variance must be a number or an array of numbers!');
 
 let mean = Math.mean(dataset); // Find the mean
 
 // For each value, find the difference from the mean
 // For each difference, find the square value
 let squaredValue = dataset.map(e => e - mean).map(e => e * e);
 
 // The variance is the average number of these squared differences
 return squaredValue.reduce((a, b) => a + b) / squaredValue.length;
}

/**
 * Returns the standard deviation for a given dataset
 * @param {number|Array} dataset
 * @param {number} dataset A number that represents the variance of the dataset
 * @returns {number}
 */
Math.std = function std(dataset) {
 if (typeof dataset === "number") return Math.sqrt(dataset); // We assume that number is the variance
 // else if dataset is an array
 else if (dataset.constructor.name !== "Array") throw new Error('Type of parameter provided to Math.std must be a number or an array of numbers!');
 
 // The standard deviation is the square root of the variance
 return Math.sqrt(Math.variance(dataset));
}