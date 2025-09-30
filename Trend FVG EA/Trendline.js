// TODO: This implementation filters out bad swings, 
// builds new good swings and 
// continues searching for extension.

TrendLineResults GetTrendLine(SwingPoint & prices[]) {
 TrendLineResults best;
 best.success = false;
 best.R_squared = 0;
 
 int n = ArraySize(prices);
 if (n < 3) return best;
 
 int start = n - 1; // latest swing
 double minR2 = 0.7;
 int maxSkips = 2; // allow a couple of bad points
 
 // progressively expand
 for (int left = start - 2; left >= 0; left--) {
  int skips = 0;
  double subPrices[];
  int validIdx[];
  
  // build slice
  for (int i = left; i <= start; i++) {
   subPrices[ArraySize(subPrices)] = prices[i].price;
   validIdx[ArraySize(validIdx)] = i;
  }
  
  TrendLineResults r = CalculateRegression(subPrices);
  if (!r.success) continue;
  
  // if R² is weak → attempt skipping
  if (r.R_squared < minR2) {
   // try excluding the worst outlier
   int worstIdx = FindWorstResidual(prices, validIdx, r);
   if (worstIdx >= 0) {
    skips++;
    if (skips > maxSkips) break; // too messy
    
    // rebuild without outlier
    double subPrices2[];
    int newIdx[];
    for (int k = 0; k < ArraySize(validIdx); k++) {
     if (validIdx[k] == worstIdx) continue;
     subPrices2[ArraySize(subPrices2)] = prices[validIdx[k]].price;
     newIdx[ArraySize(newIdx)] = validIdx[k];
    }
    
    r = CalculateRegression(subPrices2);
   }
  }
  
  // if improved, keep
  if (r.R_squared > best.R_squared) {
   best = r;
   best.startIndex = left;
   best.endIndex = start;
   best.validIdx = validIdx; // cluster reference
  }
 }
 
 return best;
}

int FindWorstResidual(SwingPoint & prices[], int & idx[], TrendLineResults & r) {
 double maxErr = 0;
 int worst = -1;
 for (int j = 0; j < ArraySize(idx); j++) {
  double predicted = r.slope * j + r.intercept;
  double err = MathAbs(prices[idx[j]].price - predicted);
  if (err > maxErr) {
   maxErr = err;
   worst = idx[j];
  }
 }
 return worst;
}

enum BOS_Type { BOS_NONE, BOS_CONTINUATION, BOS_REVERSAL };

struct BOS_Result {
 BOS_Type type;
 SwingPoint brokenSwing;
 bool success;
};

// Detect BOS with swing recording
BOS_Result DetectBOS(SwingPoint & gHs[], SwingPoint & gLs[], TrendDirection dir, double currentPrice) {
 BOS_Result result;
 result.type = BOS_NONE;
 result.success = false;
 
 // Ensure enough swings
 int nHighs = ArraySize(gHs);
 int nLows = ArraySize(gLs);
 if (nHighs < 2 || nLows < 2) return result;
 
 if (dir == TREND_UP) {
  // --- Continuation BOS: last swing high taken out ---
  for (int i = nHighs - 1; i >= 0; i--) {
   if (currentPrice > gHs[i].price) {
    result.type = BOS_CONTINUATION;
    result.brokenSwing = gHs[i];
    result.success = true;
    break;
   }
  }
  
  // --- Reversal BOS: last swing low violated ---
  for (int i = nLows - 1; i >= 0; i--) {
   if (currentPrice < gLs[i].price) {
    result.type = BOS_REVERSAL;
    result.brokenSwing = gLs[i];
    result.success = true;
    break;
   }
  }
 } else if (dir == TREND_DOWN) {
  // --- Continuation BOS: last swing low taken out ---
  for (int i = nLows - 1; i >= 0; i--) {
   if (currentPrice < gLs[i].price) {
    result.type = BOS_CONTINUATION;
    result.brokenSwing = gLs[i];
    result.success = true;
    break;
   }
  }
  
  // --- Reversal BOS: last swing high violated ---
  for (int i = nHighs - 1; i >= 0; i--) {
   if (currentPrice > gHs[i].price) {
    result.type = BOS_REVERSAL;
    result.brokenSwing = gHs[i];
    result.success = true;
    break;
   }
  }
 }
 
 return result;
}

bool DetectBOS_BySwings(MqlRates & rates[],
 SwingPoint & gHs[], SwingPoint & gLs[],
 bool isUptrend, double slopeThreshold) {
 int nHighs = ArraySize(gHs);
 int nLows = ArraySize(gLs);
 if (nHighs < 1 || nLows < 1) return false;
 
 // --- continuation BOS case (flat slope + break)
 if (IsFlatSlope(rates, slopeThreshold)) {
  if (isUptrend) {
   int idx = nHighs - 1;
   if (rates[1].high > gHs[idx].price && lastBOS.swingIndex != idx) {
    lastBOS.type = BOS_CONTINUATION;
    lastBOS.timeBroken = rates[1].time;
    lastBOS.priceBroken = gHs[idx].price;
    lastBOS.swingIndex = idx;
    lastBOS.isHigh = true;
    return true;
   }
  } else {
   int idx = nLows - 1;
   if (rates[1].low < gLs[idx].price && lastBOS.swingIndex != idx) {
    lastBOS.type = BOS_CONTINUATION;
    lastBOS.timeBroken = rates[1].time;
    lastBOS.priceBroken = gLs[idx].price;
    lastBOS.swingIndex = idx;
    lastBOS.isHigh = false;
    return true;
   }
  }
 }
 
 // --- reversal BOS case (strong break opposite trend)
 if (isUptrend) {
  int idx = nLows - 1;
  if (rates[1].low < gLs[idx].price && lastBOS.swingIndex != idx) {
   lastBOS.type = BOS_REVERSAL;
   lastBOS.timeBroken = rates[1].time;
   lastBOS.priceBroken = gLs[idx].price;
   lastBOS.swingIndex = idx;
   lastBOS.isHigh = false;
   return true;
  }
 } else {
  int idx = nHighs - 1;
  if (rates[1].high > gHs[idx].price && lastBOS.swingIndex != idx) {
   lastBOS.type = BOS_REVERSAL;
   lastBOS.timeBroken = rates[1].time;
   lastBOS.priceBroken = gHs[idx].price;
   lastBOS.swingIndex = idx;
   lastBOS.isHigh = true;
   return true;
  }
 }
 
 return false; // no new BOS
}






`
The core of the algorithm will iterate through your array of swing points, but instead of checking all possible combinations, it will progressively grow the potential trendline. 

Here's a conceptual breakdown:
Iterate and initialize: Start a loop through your array of swing points. For each point P1, begin a new potential trendline.

Add a second point: In a nested loop, select a second point P2 that occurs after P1. These two points define a temporary trendline. Calculate its initial R-squared value, which will be 1.0 since two points always have a perfect fit.

Progressively add points and check R-squared: In a third nested loop, iterate through subsequent points P3, P4, etc. For each new point, add it to the group and recalculate the R-squared value for the entire set of points.

Threshold check: This is where you implement the "more points with a fair error value" logic. Introduce a threshold value (e.g., 0.95 for R-squared). If adding a new point causes the R-squared value to drop below this threshold, stop adding points to that particular group. The previous set of points is the best fit for that potential trendline.

Compare and store: After each "group" of points has been fully evaluated (either by reaching the end of the array or by the R-squared dropping below the threshold), compare its R-squared value to the best R-squared found so far. 
If the current group's R-squared is higher, and it consists of a "better" number of points (based on a tie-breaker rule you can define), store the indices of these points and their R-squared value as the new best match.
`
void FindBestTrendline() {
 double bestRSquared = -1.0;
 int bestPointsCount = 0;
 int bestStartIndex = -1;
 int bestEndIndex = -1;
 
 // Iterate through all possible starting points
 for (int i = 1; i < ArraySize(m_swing_points) - 2; i++) {
  // Iterate through all possible second points
  for (int j = i + 1; j < ArraySize(m_swing_points) - 1; j++) {
   // Temporary arrays for the current group of points
   CArrayObj * currentPoints = new CArrayObj();
   currentPoints.Add(GetPointer(m_swing_points[i]));
   currentPoints.Add(GetPointer(m_swing_points[j]));
   
   double currentRSquared = 1.0; // R-squared is 1.0 for two points
   
   // Progressively add more points and check the R-squared
   for (int k = j + 1; k < ArraySize(m_swing_points); k++) {
    currentPoints.Add(GetPointer(m_swing_points[k]));
    
    // Calculate new R-squared
    currentRSquared = CalculateRSquared(currentPoints);
    
    // Break if the R-squared drops below the threshold
    if (currentRSquared < RSquaredThreshold) {
     break;
    }
   }
   
   // Check if this group is better than the current best
   int currentPointsCount = currentPoints.Total();
   if (currentRSquared > bestRSquared) {
    bestRSquared = currentRSquared;
    bestPointsCount = currentPointsCount;
    bestStartIndex = i;
    bestEndIndex = i + (currentPointsCount - 1);
   }
   // Optional: Tie-breaker for same R-squared values, choose more points
   else if (currentRSquared == bestRSquared && currentPointsCount > bestPointsCount) {
    bestPointsCount = currentPointsCount;
    bestStartIndex = i;
    bestEndIndex = i + (currentPointsCount - 1);
   }
   
   delete currentPoints;
  }
 }
 
 // Print the result. You can use these indices to draw the line later.
 Print("Best Trendline Found!");
 Print("  - R-Squared Value: ", DoubleToString(bestRSquared, 4));
 Print("  - Number of points: ", bestPointsCount);
 Print("  - Start Index: ", bestStartIndex, ", End Index: ", bestEndIndex);
}
//--------------------------------------------------------------------+
// Helper function to calculate R-squared for a set of points
//--------------------------------------------------------------------+
double CalculateRSquared(CArrayObj * points) {
 // Check if there are enough points (at least 2)
 if (points.Total() < 2) return 0.0;
 
 // Arrays for linear regression calculation
 double x[], y[];
 
 // Populate x and y arrays from the CArrayObj
 ArrayResize(x, points.Total());
 ArrayResize(y, points.Total());
 for (int i = 0; i < points.Total(); i++) {
  SwingPoint * p = (SwingPoint * ) points.At(i);
  x[i] = p.time; // Use time as the independent variable (x)
  y[i] = p.price; // Use price as the dependent variable (y)
 }
 
 // Calculate slope and intercept of the regression line
 double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
 int n = points.Total();
 
 for (int i = 0; i < n; i++) {
  sumX += x[i];
  sumY += y[i];
  sumXY += x[i] * y[i];
  sumX2 += x[i] * x[i];
 }
 
 double slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
 double intercept = (sumY - slope * sumX) / n;
 
 // Calculate R-squared
 double SS_total = 0, SS_residual = 0;
 double avgY = sumY / n;
 
 for (int i = 0; i < n; i++) {
  // Total sum of squares
  SS_total += MathPow(y[i] - avgY, 2);
  
  // Residual sum of squares (error)
  double predictedY = slope * x[i] + intercept;
  SS_residual += MathPow(y[i] - predictedY, 2);
 }
 
 // Handle division by zero
 if (SS_total == 0) return 1.0;
 
 return 1 - (SS_residual / SS_total);
}

`
How to Implement the Reversed Check
You don't need to rebuild the entire algorithm. You can simply reuse your CalculateRSquared function within a new loop structure.

Identify the best-fit group: After your initial forward loops complete, you have the bestStartIndex and bestEndIndex of the highest R-squared group.

Iterate backward: Create a new loop that starts from bestStartIndex and moves backward in your m_swing_points array.

Add early points: In this loop, add each earlier point to a temporary array containing the original best-fit group.

Re-calculate and compare: For each new point added, recalculate the R-squared for the entire extended group. If the R-squared remains above your RSquaredThreshold, continue adding points. If it drops, stop.

Store the final result: The new, longer group of points is your final result. Compare its R-squared and point count to the original best. This ensures you only choose the new group if it's genuinely better.
`

void FindBestTrendline() {
 // ... (Your existing forward-scan code to find bestRSquared, bestStartIndex, bestEndIndex) ...
 
 //------------------------------------------------------------------+
 // REVERSED CHECK - Step 1: Initialize temporary array with the best points found
 //------------------------------------------------------------------+
 CArrayObj * finalPoints = new CArrayObj();
 for (int i = bestStartIndex; i <= bestEndIndex; i++) {
  finalPoints.Add(GetPointer(m_swing_points[i]));
 }
 
 // Get the current best R-squared and point count as a baseline
 double finalRSquared = bestRSquared;
 int finalPointsCount = bestPointsCount;
 
 //------------------------------------------------------------------+
 // Step 2: Loop backward from the start of the current best group
 //------------------------------------------------------------------+
 for (int i = bestStartIndex - 1; i >= 0; i--) {
  // Add the new, earlier point to the front of the array
  finalPoints.Insert(0, GetPointer(m_swing_points[i]));
  
  // Recalculate R-squared for the extended group
  double newRSquared = CalculateRSquared(finalPoints);
  
  // Check if the R-squared is still within the threshold
  if (newRSquared >= RSquaredThreshold) {
   // Update the final best values if the extended group is valid
   finalRSquared = newRSquared;
   finalPointsCount = finalPoints.Total();
  } else {
   // If the threshold is broken, remove the last added point and break the loop
   finalPoints.Remove(0);
   break;
  }
 }
 
 // Print the final result
 Print("Final Best Trendline Found!");
 Print("  - R-Squared Value: ", DoubleToString(finalRSquared, 4));
 Print("  - Number of points: ", finalPointsCount);
 // You can now use the finalPoints array to get the start/end times
 
 delete finalPoints;
}




//+------------------------------------------------------------------+
//|                                                   Trendline.mq5 |
//+------------------------------------------------------------------+
#property indicator_separate_window

// Assume m_swing_points[] is already populated
struct SwingPoint {
 datetime time;
 double price;
};

extern double RSquaredThreshold = 0.95; // User-defined threshold

// Assuming m_swing_points[] is a global array of SwingPoint structs
SwingPoint m_swing_points[];

void OnStart() {
 
 // Check if there are enough swing points to analyze
 if (ArraySize(m_swing_points) < 3) {
  Print("Not enough swing points to find a trendline.");
  return;
 }
 
 FindBestTrendline();
}

//--------------------------------------------------------------------+
// Function to find the best-fitting trendline using R-squared
//--------------------------------------------------------------------+
void FindBestTrendline() {
 double bestRSquared = -1.0;
 int bestPointsCount = 0;
 int bestStartIndex = -1;
 int bestEndIndex = -1;
 
 // Iterate through all possible starting points
 for (int i = 0; i < ArraySize(m_swing_points) - 2; i++) {
  // Iterate through all possible second points
  for (int j = i + 1; j < ArraySize(m_swing_points) - 1; j++) {
   
   // Temporary array for the current group of points
   SwingPoint currentPoints[];
   
   // Start with the first two points
   ArrayResize(currentPoints, 2);
   currentPoints[0] = m_swing_points[i];
   currentPoints[1] = m_swing_points[j];
   
   double currentRSquared = 1.0; // R-squared is 1.0 for two points
   
   // Progressively add more points and check the R-squared
   for (int k = j + 1; k < ArraySize(m_swing_points); k++) {
    
    // Add the new point to the temporary array
    int newSize = ArraySize(currentPoints) + 1;
    ArrayResize(currentPoints, newSize);
    currentPoints[newSize - 1] = m_swing_points[k];
    
    // Calculate new R-squared
    currentRSquared = CalculateRSquared(currentPoints);
    
    // Break if the R-squared drops below the threshold
    if (currentRSquared < RSquaredThreshold) {
     // Remove the last point that broke the threshold
     ArrayResize(currentPoints, newSize - 1);
     break;
    }
   }
   
   // Check if this group is better than the current best
   int currentPointsCount = ArraySize(currentPoints);
   if (currentRSquared > bestRSquared) {
    bestRSquared = currentRSquared;
    bestPointsCount = currentPointsCount;
    bestStartIndex = i;
    bestEndIndex = j + (currentPointsCount - 2); // Calculate the end index correctly
   }
   // Optional: Tie-breaker for same R-squared values, choose more points
   else if (currentRSquared == bestRSquared && currentPointsCount > bestPointsCount) {
    bestPointsCount = currentPointsCount;
    bestStartIndex = i;
    bestEndIndex = j + (currentPointsCount - 2);
   }
  }
 }
 
 //------------------------------------------------------------------+
 // REVERSED CHECK - Step 1: Initialize temporary array with the best points found
 //------------------------------------------------------------------+
 SwingPoint finalPoints[];
 ArrayResize(finalPoints, bestPointsCount);
 ArrayCopy(finalPoints, m_swing_points, 0, bestStartIndex, bestPointsCount);
 
 // Get the current best R-squared and point count as a baseline
 double finalRSquared = bestRSquared;
 int finalPointsCount = bestPointsCount;
 
 //------------------------------------------------------------------+
 // Step 2: Loop backward from the start of the current best group
 //------------------------------------------------------------------+
 for (int i = bestStartIndex - 1; i >= 0; i--) {
  // Prepend the new, earlier point to the array
  int newSize = ArraySize(finalPoints) + 1;
  ArrayResize(finalPoints, newSize);
  ArrayCopy(finalPoints, finalPoints, 1, 0, newSize - 1);
  finalPoints[0] = m_swing_points[i];
  
  // Recalculate R-squared for the extended group
  double newRSquared = CalculateRSquared(finalPoints);
  
  // Check if the R-squared is still within the threshold
  if (newRSquared >= RSquaredThreshold) {
   // Update the final best values if the extended group is valid
   finalRSquared = newRSquared;
   finalPointsCount = ArraySize(finalPoints);
  } else {
   // If the threshold is broken, remove the last added point and break the loop
   ArrayResize(finalPoints, ArraySize(finalPoints) - 1);
   break;
  }
 }
 
 // Print the final result
 Print("Final Best Trendline Found!");
 Print("  - R-Squared Value: ", DoubleToString(finalRSquared, 4));
 Print("  - Number of points: ", finalPointsCount);
 Print("  - Start Point Time: ", TimeToString(finalPoints[0].time));
 Print("  - End Point Time: ", TimeToString(finalPoints[ArraySize(finalPoints) - 1].time));
}

//--------------------------------------------------------------------+
// Helper function to calculate R-squared for a set of points
//--------------------------------------------------------------------+
double CalculateRSquared(const SwingPoint & points[]) {
 // Check if there are enough points (at least 2)
 int n = ArraySize(points);
 if (n < 2) return 0.0;
 
 // Arrays for linear regression calculation
 double x[], y[];
 ArrayResize(x, n);
 ArrayResize(y, n);
 
 // Populate x and y arrays from the input array
 for (int i = 0; i < n; i++) {
  x[i] = points[i].time; // Use time as the independent variable (x)
  y[i] = points[i].price; // Use price as the dependent variable (y)
 }
 
 // Calculate slope and intercept of the regression line
 double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
 
 for (int i = 0; i < n; i++) {
  sumX += x[i];
  sumY += y[i];
  sumXY += x[i] * y[i];
  sumX2 += x[i] * x[i];
 }
 
 double slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
 double intercept = (sumY - slope * sumX) / n;
 
 // Calculate R-squared
 double SS_total = 0, SS_residual = 0;
 double avgY = sumY / n;
 
 for (int i = 0; i < n; i++) {
  // Total sum of squares
  SS_total += MathPow(y[i] - avgY, 2);
  
  // Residual sum of squares (error)
  double predictedY = slope * x[i] + intercept;
  SS_residual += MathPow(y[i] - predictedY, 2);
 }
 
 // Handle division by zero
 if (SS_total == 0) return 1.0;
 
 return 1 - (SS_residual / SS_total);
}