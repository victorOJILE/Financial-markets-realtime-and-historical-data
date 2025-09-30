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
  
  // Sort points by time to ensure correct order
  swingPoints.sort((a, b) => a.time - b.time);
  
  let allValidCandidates = [];
  
  // Run the custom random forest logic
  for (let i = 0; i < this.numTrees; i++) {
   // 1. Random Subset Selection
   const numPointsToSelect = Math.floor(Math.random() * (swingPoints.length - this.minPoints + 1)) + this.minPoints;
   const selectedIndices = new Set();
   while (selectedIndices.size < numPointsToSelect) selectedIndices.add(Math.floor(Math.random() * swingPoints.length));
   
   const selectedPoints = Array.from(selectedIndices).sort((a, b) => a - b).map(idx => swingPoints[idx]);
   
   // Perform linear regression
   const regressionResult = this._calculateLinearRegression(selectedPoints);
   
   if (regressionResult && regressionResult.r_squared >= this.minR_squared) {
    // 2. Backward Loop Residual Check
    if (this._checkResiduals(regressionResult, swingPoints, selectedPoints)) {
     allValidCandidates.push({
      r_squared: regressionResult.r_squared,
      points: selectedPoints,
      slope: regressionResult.slope,
      intercept: regressionResult.intercept
     });
    }
   }
  }
  
  // 3. Ensemble and Voting
  if (allValidCandidates.length === 0) {
   return null;
  }
  
  return allValidCandidates.reduce((best, current) => current.r_squared > best.r_squared ? current : best);
 }
 
 _calculateLinearRegression(points) {
  const n = points.length;
  if (n < 2) return null;
  
  let sum_x = 0,
   sum_y = 0,
   sum_xy = 0,
   sum_x2 = 0;
  points.forEach(p => {
   sum_x += p.time;
   sum_y += p.price;
   sum_xy += p.time * p.price;
   sum_x2 += p.time * p.time;
  });
  
  const denominator = n * sum_x2 - sum_x * sum_x;
  if (Math.abs(denominator) < 1e-6) return { r_squared: 0, slope: 0, intercept: sum_y / n }; // Horizontal line
  
  const slope = (n * sum_xy - sum_x * sum_y) / denominator;
  const intercept = (sum_y - slope * sum_x) / n;
  
  const y_mean = sum_y / n;
  let ss_total = 0,
   ss_residual = 0;
  points.forEach(p => {
   const predicted_y = slope * p.time + intercept;
   ss_total += Math.pow(p.price - y_mean, 2);
   ss_residual += Math.pow(p.price - predicted_y, 2);
  });
  
  const r_squared = (Math.abs(ss_total) < 1e-6) ? 1 : (1 - (ss_residual / ss_total));
  
  return { r_squared, slope, intercept };
 }
 
 _checkResiduals(trend, allPoints, trendPoints) {
  const earliestTrendTime = trendPoints[0].time;
  const earlierPoints = allPoints.filter(p => p.time < earliestTrendTime);
  
  const pointsToCheck = earlierPoints.slice(-this.residualCheckPoints);
  
  if (pointsToCheck.length === 0) return true;
  
  const isUptrend = trend.slope > 0;
  let validResidualsCount = 0;
  
  pointsToCheck.forEach(p => {
   const predictedY = trend.slope * p.time + trend.intercept;
   const residual = p.price - predictedY;
   
   if ((isUptrend && residual > 0) || (!isUptrend && residual < 0)) {
    validResidualsCount++;
   }
  });
  
  return (validResidualsCount / pointsToCheck.length) >= this.minResidualCheckPct;
 }
}

function runAlgorithm() {
 const params = {
  minR_squared: 0.99,
  minPoints: 3,
  numTrees: 100,
  residualCheckPoints: 10,
  minResidualCheckPct: 0.8
 };
 
 const allPoints = generateSwingPoints('uptrend'); // You can change 'uptrend' to 'downtrend'
 const trendFinder = new CustomTrendFinder(params);
 
 console.log('Running algorithm...');
 const startTime = performance.now();
 
 // Use a setTimeout to allow the UI to update with the "Running" message
 setTimeout(() => {
  const foundTrend = trendFinder.findTrend(allPoints);
  const endTime = performance.now();
  const elapsedTime = (endTime - startTime).toFixed(2);
  
  if (foundTrend) {
   console.log(`Algorithm finished in ${elapsedTime}ms. ✅ Trend found and validated!`);
   console.log('R_squared: ' + foundTrend.r_squared.toFixed(4));
   console.log('Points used: ' + foundTrend.points.map(p => `(${p.time.toFixed(2)}, ${p.price.toFixed(2)})`).join(', '));
  } else {
   console.log(`Algorithm finished in ${elapsedTime}ms. ❌ No valid trend line found.`);
  }
 }, 10);
}

runAlgorithm();