class Node {
 constructor(featureIndex = null, threshold = null, left = null, right = null, value = null) {
  this.featureIndex = featureIndex;
  this.threshold = threshold;
  this.left = left;
  this.right = right;
  this.value = value;
 }
}

class DecisionTree {
 constructor({ maxDepth = null, minSamplesSplit = 2, criterion = "entropy" } = {}) {
  this.maxDepth = maxDepth;
  this.minSamplesSplit = minSamplesSplit;
  this.criterion = criterion.toLowerCase();
  this.root = null;
  
  // Custom function to handle base-2 logarithm in JS
  if (typeof Math.log2 !== 'function') {
   Math.log2 = function(x) {
    return Math.log(x) * Math.LOG2E;
   };
  }
 }
 
 fit(X, y) {
  this.root = this._growTree(X, y);
 }
 
 predict(X) {
  return X.map(x => this._traverseTree(x, this.root));
 }
 
 _getUniqueValues(arr) {
  return Array.from(new Set(arr));
 }
 
 _getCounts(arr) {
  const counts = new Map();
  for (const item of arr) {
   counts.set(item, (counts.get(item) || 0) + 1);
  }
  return counts;
 }
 
 _filterDataByIndices(X, y, indices) {
  const newX = [];
  const newY = [];
  for (const idx of indices) {
   newX.push(X[idx]);
   newY.push(y[idx]);
  }
  return { X: newX, y: newY };
 }
 
 _getIndicesWhere(arr, predicate) {
  return arr.map((val, idx) => predicate(val, idx) ? idx : -1).filter(idx => idx !== -1);
 }
 
 _growTree(X, y, depth = 0) {
  const nSamples = X.length;
  if (nSamples === 0) return new Node(value = null);
  const nFeatures = X[0] ? X[0].length : 0;
  const nLabels = this._getUniqueValues(y).length;
  
  // Stopping criteria: pure node, max depth, or too few samples.
  if (
   (this.maxDepth !== null && depth >= this.maxDepth) ||
   nLabels === 1 ||
   nSamples < this.minSamplesSplit ||
   nFeatures === 0
  ) {
   const leafValue = this._mostCommonLabel(y);
   return new Node(null, null, null, null, leafValue);
  }
  
  // Find the best split
  const bestSplit = this._findBestSplit(X, y);
  
  if (!bestSplit || bestSplit.infoGain <= 0) {
   const leafValue = this._mostCommonLabel(y);
   return new Node(null, null, null, null, leafValue);
  }
  
  // Split the data
  const { featureIdx, threshold, leftIdxs, rightIdxs } = bestSplit;
  const { X: XLeft, y: yLeft } = this._filterDataByIndices(X, y, leftIdxs);
  const { X: XRight, y: yRight } = this._filterDataByIndices(X, y, rightIdxs);
  
  // Recursively grow the children
  const leftChild = this._growTree(XLeft, yLeft, depth + 1);
  const rightChild = this._growTree(XRight, yRight, depth + 1);
  
  return new Node(featureIdx, threshold, leftChild, rightChild, null);
 }
 
 _findBestSplit(X, y) {
  let bestGain = -Infinity;
  let bestSplit = null;
  const nFeatures = X[0].length;
  
  // Iterate through all features
  for (let featureIdx = 0; featureIdx < nFeatures; featureIdx++) {
   const featureValues = X.map(row => row[featureIdx]);
   const categories = this._getUniqueValues(featureValues);
   
   // Iterate through every unique category value as a potential split threshold
   for (const cat of categories) {
    // Split: (featureValue == cat) vs (featureValue != cat)
    const leftIdxs = this._getIndicesWhere(featureValues, val => val === cat);
    const rightIdxs = this._getIndicesWhere(featureValues, val => val !== cat);
    
    // Skip trivial splits (where all data goes to one side)
    if (leftIdxs.length === 0 || rightIdxs.length === 0) continue;
    
    const infoGain = this._calculateInformationGain(y, leftIdxs, rightIdxs);
    
    if (infoGain > bestGain) {
     bestGain = infoGain;
     bestSplit = {
      featureIdx: featureIdx,
      threshold: cat, // Category value
      infoGain: bestGain,
      leftIdxs: leftIdxs,
      rightIdxs: rightIdxs
     };
    }
   }
  }
  return bestSplit;
 }
 
 /**
  * Calculates the information gain for a given split.
  */
 _calculateInformationGain(y, leftIdxs, rightIdxs) {
  const parentImpurity = this._impurity(y);
  
  const yLeft = this._filterDataByIndices([], y, leftIdxs).y;
  const yRight = this._filterDataByIndices([], y, rightIdxs).y;
  
  // This check is redundant due to the check in _findBestSplit, but kept for robustness
  if (yLeft.length === 0 || yRight.length === 0) {
   return 0;
  }
  
  const n = y.length;
  const nLeft = yLeft.length;
  const nRight = yRight.length;
  
  const weightedChildImpurity = (nLeft / n) * this._impurity(yLeft) +
   (nRight / n) * this._impurity(yRight);
  
  return parentImpurity - weightedChildImpurity;
 }
 
 /**
  * Calculates the impurity (Gini or Entropy) of a set of labels.
  */
 _impurity(y) {
  const n = y.length;
  if (n === 0) {
   return 0;
  }
  
  const counts = this._getCounts(y);
  const probabilities = Array.from(counts.values()).map(count => count / n);
  
  if (this.criterion === "gini") {
   // Gini Impurity: 1 - sum(p^2)
   return 1 - probabilities.reduce((sum, p) => sum + p ** 2, 0);
  } else { // entropy
   // Entropy: -sum(p * log2(p))
   return -probabilities.reduce((sum, p) => {
    if (p > 0) return sum + p * Math.log2(p);
    return sum;
   }, 0);
  }
 }
 
 /**
  * Traverses the tree to predict the label for a single data point.
  */
 _traverseTree(x, node) {
  // Base case: leaf node
  if (node.value !== null) return node.value;
  
  const featureValue = x[node.featureIndex];
  
  // Always treat as categorical split: check for equality
  if (featureValue === node.threshold) {
   return this._traverseTree(x, node.left);
  } else {
   return this._traverseTree(x, node.right);
  }
 }
 
 /**
  * Finds the most common label in the array.
  */
 _mostCommonLabel(y) {
  if (y.length === 0) return null;
  
  const counts = this._getCounts(y);
  let maxCount = -1;
  let mostCommon = null;
  
  for (const [label, count] of counts.entries()) {
   if (count > maxCount) {
    maxCount = count;
    mostCommon = label;
   }
  }
  return mostCommon;
 }
}

// --- Example Usage ---

/*
 * Example Dataset: Play Tennis (all features are categorical strings)
 * Features: Outlook, Temp, Humidity, Wind
 * Target Y: Play (Yes/No)
 *
const X_train = [
 ["Sunny", "Hot", "High", "Weak"],
 ["Sunny", "Hot", "High", "Strong"],
 ["Overcast", "Hot", "High", "Weak"],
 ["Rain", "Mild", "High", "Weak"],
 ["Rain", "Cool", "Normal", "Weak"],
 ["Rain", "Cool", "Normal", "Strong"],
 ["Overcast", "Cool", "Normal", "Strong"],
 ["Sunny", "Mild", "High", "Weak"],
 ["Sunny", "Cool", "Normal", "Weak"],
 ["Rain", "Mild", "Normal", "Weak"],
 ["Sunny", "Mild", "Normal", "Strong"],
 ["Overcast", "Mild", "High", "Strong"],
 ["Overcast", "Hot", "Normal", "Weak"],
 ["Rain", "Mild", "High", "Strong"]
];
const y_train = ["No", "No", "Yes", "Yes", "Yes", "No", "Yes", "No", "Yes", "Yes", "Yes", "Yes", "Yes", "No"];

const classifier = new DecisionTree({
 maxDepth: 4,
 criterion: "entropy",
});

classifier.fit(X_train, y_train);

// Test Data
const X_test = [
 ["Sunny", "Hot", "Normal", "Weak"], // Expected: Yes
 ["Rain", "Cool", "High", "Strong"], // Expected: No
 ["Overcast", "Hot", "High", "Weak"] // Expected: Yes
];

const predictions = classifier.predict(X_test);

console.log("--- Decision Tree Classifier Results ---");
console.log("Predictions:", predictions); // Output: [ 'Yes', 'No', 'Yes' ]

// Helper function to show the tree structure
function printTree(node, spacing = "") {
 if (node.value !== null) {
  console.log(spacing + `[LEAF] Predict: ${node.value}`);
  return;
 }
 const featureName = `Feature ${node.featureIndex}`;
 console.log(spacing + `[${featureName} IS "${node.threshold}"]`);
 
 console.log(spacing + '  --> True:');
 printTree(node.left, spacing + '    ');
 
 console.log(spacing + '  --> False (All Others):');
 printTree(node.right, spacing + '    ');
}

console.log("\n--- Generated Tree Structure (Depth 4) ---");
console.log(classifier.root);

*/



const TRADING_DATA = [
    // --- 1. STRONG BULLISH (Confirmed SMT + Bullish Liquidity/Bias) [LABEL: 1] ---
    [0, 0, 2, 0, 0, -1, 0, 0, -1, -1, -1, 0, 0, 0, 1],
    [0, 0, 1, 0, 0, -1, 0, 0, -1, -1, -1, 0, 0, 0, 1],
    [0, -1, 2, 0, 0, -1, -1, -1, -1, -1, 0, 1, 0, 0, 1], // OB broken Bullish
    [0, 0, 1, 0, -1, 0, -1, -1, -1, -1, -1, 0, 0, 0, 1], // Approaching Bullish FVG
    [0, 0, 2, 0, 0, -1, -1, -1, -1, 0, -1, 0, 0, 0, 1], // In Bullish OB
    [0, 0, 1, 0, 0, -1, -1, 0, -1, -1, -1, 0, 0, 0, 1], // Bullish FVG mitigated
    [0, 0, 2, 0, 0, -1, 0, -1, -1, -1, -1, 0, 0, 0, 1], // In Bullish FVG
    [0, -1, 1, 0, 0, 0, -1, -1, -1, -1, -1, 0, 0, 0, 1],
    [0, 0, 2, 0, 0, -1, -1, 0, -1, -1, -1, 0, 0, 0, 1],
    [0, 0, 1, 0, 0, -1, 0, -1, -1, -1, -1, 0, 0, 0, 1],
    [0, 0, 2, 0, 0, -1, 0, -1, -1, -1, -1, 0, 0, 0, 1],
    [0, 0, 1, 0, 0, -1, -1, -1, -1, 0, -1, 0, 0, 0, 1],
    [0, 0, 2, 0, 0, -1, -1, 0, -1, -1, -1, 0, 0, 0, 1],
    [0, -1, 1, 0, 0, -1, 0, -1, -1, -1, -1, 0, 0, 0, 1],
    [0, 0, 2, 0, 0, 0, -1, -1, -1, -1, -1, 0, 0, 0, 1],
    [0, 0, 1, 0, 0, -1, 0, -1, -1, -1, -1, 0, 0, 0, 1],
    [0, 0, 2, 0, 0, -1, 0, -1, -1, -1, -1, 0, 0, 0, 1],
    [0, 0, 1, 0, 0, 0, -1, -1, -1, -1, -1, 0, 0, 0, 1],
    [0, 0, 2, 0, 0, -1, 0, -1, -1, -1, -1, 0, 0, 0, 1],
    [0, 0, 1, 0, 0, -1, -1, 0, -1, -1, -1, 0, 0, 0, 1],

    // --- 2. STRONG BEARISH (Confirmed SMT + Bearish Liquidity/Bias) [LABEL: -1] ---
    [1, 1, 2, 1, 1, -1, 1, -1, -1, -1, -1, 0, 1, 1, -1], // In Bearish FVG
    [1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 0, 1, 1, -1], // Approaching Bearish FVG
    [1, 1, 2, 1, 1, -1, -1, -1, -1, 1, -1, 0, 1, 1, -1], // In Bearish OB
    [1, 1, 1, 1, 1, -1, -1, 1, -1, -1, -1, 0, 1, 1, -1], // Bearish FVG mitigated
    [1, 1, 2, 1, 1, -1, -1, -1, -1, -1, 1, 0, 1, 1, -1], // OB broken Bearish
    [1, 1, 1, 1, 1, -1, 1, -1, -1, -1, -1, 0, 1, 1, -1],
    [1, 1, 2, 1, 1, 1, -1, -1, -1, -1, -1, 0, 1, 1, -1],
    [1, 1, 1, 1, 1, -1, 1, -1, -1, -1, -1, 0, 1, 1, -1],
    [1, 1, 2, 1, 1, -1, -1, 1, -1, -1, -1, 0, 1, 1, -1],
    [1, 1, 1, 1, 1, -1, 1, -1, -1, -1, -1, 0, 1, 1, -1],
    [1, 1, 2, 1, 1, -1, 1, -1, -1, -1, -1, 0, 1, 1, -1],
    [1, 1, 1, 1, 1, -1, -1, -1, -1, 1, -1, 0, 1, 1, -1],
    [1, 1, 2, 1, 1, -1, 1, -1, -1, -1, -1, 0, 1, 1, -1],
    [1, 1, 1, 1, 1, -1, -1, 1, -1, -1, -1, 0, 1, 1, -1],
    [1, 1, 2, 1, 1, -1, 1, -1, -1, -1, -1, 0, 1, 1, -1],
    [1, 1, 1, 1, 1, -1, -1, -1, -1, 1, -1, 0, 1, 1, -1],
    [1, 1, 2, 1, 1, -1, 1, -1, -1, -1, -1, 0, 1, 1, -1],
    [1, 1, 1, 1, 1, -1, -1, 1, -1, -1, -1, 0, 1, 1, -1],
    [1, 1, 2, 1, 1, -1, 1, -1, -1, -1, -1, 0, 1, 1, -1],
    [1, 1, 1, 1, 1, -1, -1, -1, -1, 1, -1, 0, 1, 1, -1],

    // --- 3. HOLD SCENARIOS (SMT Detected but NOT Confirmed / Conflicting Signals) [LABEL: 0] ---
    // Divergence detected, waiting for FVG/OB or Engulfing confirmation.
    [0, 0, 1, 0, -1, 0, -1, -1, -1, -1, -1, 0, 0, -1, 0], // Bullish SMT Detected, approaching FVG demand (WAIT)
    [1, 1, 2, 1, -1, 1, -1, -1, -1, -1, -1, 0, 1, -1, 0], // Bearish SMT Detected, approaching FVG supply (WAIT)
    [0, -1, 1, 0, -1, -1, -1, -1, -1, 0, -1, 0, 0, -1, 0], // Bullish SMT Detected, in OB demand (WAIT)
    [1, 0, 2, 1, -1, -1, -1, -1, -1, 1, -1, 0, 1, -1, 0], // Bearish SMT Detected, in OB supply (WAIT)
    [0, 1, 1, 0, 1, -1, -1, -1, -1, -1, -1, 0, 0, -1, 0], // Bullish SMT Detected, but Engulfing is Bearish (WAIT)
    [1, 0, 2, 1, 0, -1, -1, -1, -1, -1, -1, 0, 1, -1, 0], // Bearish SMT Detected, but Engulfing is Bullish (WAIT)
    [-1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1, 0], // SMT Detected, No other bias (WAIT)
    [-1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, -1, 0], // SMT Detected, No other bias (WAIT)

    // Conflicting Confirmed SMT (Must be a HOLD, neutralized to -1)
    [0, 1, 1, 0, 1, -1, -1, -1, -1, -1, -1, 0, -1, -1, 0], // Contradiction: Both SMT Confirmed (WAIT)
    [0, -1, 2, 1, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0], // Bullish SMT Confirmed, but MA is Bearish (WAIT)
    [1, 0, 1, 0, 1, -1, -1, -1, -1, -1, -1, 0, 1, 1, 0], // Bearish SMT Confirmed, but MA is Bullish (WAIT)

    // --- 4. WEAKER BUYS (General Bias, No SMT Detected) [LABEL: 1] ---
    [0, 0, 2, 0, 0, 0, -1, -1, -1, -1, -1, 0, -1, -1, 1],
    [0, -1, 1, 0, 0, -1, -1, -1, -1, -1, 0, 1, -1, -1, 1], // OB broken Bullish
    [0, 0, 2, 0, 0, -1, 0, -1, -1, -1, -1, 0, -1, -1, 1],
    [0, 0, 1, 0, 0, 0, -1, -1, -1, -1, -1, 0, -1, -1, 1],
    [0, 0, 2, 0, 0, -1, 0, -1, -1, -1, -1, 0, -1, -1, 1],
    [0, 0, 1, 0, 0, -1, -1, 0, -1, -1, -1, 0, -1, -1, 1],
    [0, 0, 2, 0, 0, -1, 0, -1, -1, -1, -1, 0, -1, -1, 1],
    [0, 0, 1, 0, 0, -1, -1, -1, -1, 0, -1, 0, -1, -1, 1],
    [0, 0, 2, 0, 0, -1, 0, -1, -1, -1, -1, 0, -1, -1, 1],
    [0, 0, 1, 0, 0, 0, -1, -1, -1, -1, -1, 0, -1, -1, 1],
    [0, 0, 2, 0, 0, -1, 0, -1, -1, -1, -1, 0, -1, -1, 1],
    [0, 0, 1, 0, 0, -1, -1, 0, -1, -1, -1, 0, -1, -1, 1],
    [0, 0, 2, 0, 0, -1, 0, -1, -1, -1, -1, 0, -1, -1, 1],
    [0, 0, 1, 0, 0, 0, -1, -1, -1, -1, -1, 0, -1, -1, 1],
    [0, 0, 2, 0, 0, -1, 0, -1, -1, -1, -1, 0, -1, -1, 1],
    [0, 0, 1, 0, 0, -1, -1, 0, -1, -1, -1, 0, -1, -1, 1],
    [0, 0, 2, 0, 0, -1, 0, -1, -1, -1, -1, 0, -1, -1, 1],
    [0, 0, 1, 0, 0, 0, -1, -1, -1, -1, -1, 0, -1, -1, 1],
    [0, 0, 2, 0, 0, -1, 0, -1, -1, -1, -1, 0, -1, -1, 1],
    [0, 0, 1, 0, 0, -1, -1, 0, -1, -1, -1, 0, -1, -1, 1],

    // --- 5. WEAKER SELLS (General Bias, No SMT Detected) [LABEL: -1] ---
    [1, 1, 2, 1, 1, 1, -1, -1, -1, -1, -1, 0, -1, -1, -1],
    [1, 1, 1, 1, 1, -1, 1, -1, -1, -1, -1, 0, -1, -1, -1],
    [1, 1, 2, 1, 1, -1, -1, 1, -1, -1, -1, 0, -1, -1, -1],
    [1, 1, 1, 1, 1, -1, -1, -1, -1, 1, -1, 0, -1, -1, -1],
    [1, 1, 2, 1, 1, -1, 1, -1, -1, -1, -1, 0, -1, -1, -1],
    [1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 0, -1, -1, -1],
    [1, 1, 2, 1, 1, -1, 1, -1, -1, -1, -1, 0, -1, -1, -1],
    [1, 1, 1, 1, 1, -1, -1, 1, -1, -1, -1, 0, -1, -1, -1],
    [1, 1, 2, 1, 1, -1, 1, -1, -1, -1, -1, 0, -1, -1, -1],
    [1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 0, -1, -1, -1],
    [1, 1, 2, 1, 1, -1, 1, -1, -1, -1, -1, 0, -1, -1, -1],
    [1, 1, 1, 1, 1, -1, -1, 1, -1, -1, -1, 0, -1, -1, -1],
    [1, 1, 2, 1, 1, -1, 1, -1, -1, -1, -1, 0, -1, -1, -1],
    [1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 0, -1, -1, -1],
    [1, 1, 2, 1, 1, -1, 1, -1, -1, -1, -1, 0, -1, -1, -1],
    [1, 1, 1, 1, 1, -1, -1, 1, -1, -1, -1, 0, -1, -1, -1],
    [1, 1, 2, 1, 1, -1, 1, -1, -1, -1, -1, 0, -1, -1, -1],
    [1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 0, -1, -1, -1],
    [1, 1, 2, 1, 1, -1, 1, -1, -1, -1, -1, 0, -1, -1, -1],
    [1, 1, 1, 1, 1, -1, -1, 1, -1, -1, -1, 0, -1, -1, -1],

    // --- 6. ADDITIONAL HOLDS / MIXED BIAS (SMT Detection Active or Conflicting) [LABEL: 0] ---
    // Price in conflicting zones or SMT detected, but bias is mixed.
    [0, 1, 1, 0, 1, -1, 1, -1, -1, -1, -1, 0, 0, -1, 0], // Bullish SMT Detected, but In Bearish FVG (WAIT)
    [1, 0, 2, 1, 0, -1, 0, -1, -1, -1, -1, 0, 1, -1, 0], // Bearish SMT Detected, but In Bullish FVG (WAIT)
    [-1, -1, 1, 0, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1, 0],
    [-1, -1, 2, 1, -1, -1, -1, -1, -1, -1, -1, 0, 1, -1, 0],
    [0, 1, 1, 0, 1, -1, -1, 1, -1, -1, -1, 0, 0, -1, 0],
    [1, 0, 2, 1, 0, -1, -1, 0, -1, -1, -1, 0, 1, -1, 0],
    [-1, -1, 1, 0, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1, 0],
    [-1, -1, 2, 1, -1, -1, -1, -1, -1, -1, -1, 0, 1, -1, 0],
    [0, 1, 1, 0, 1, -1, -1, -1, -1, -1, -1, 0, 0, -1, 0],
    [1, 0, 2, 1, 0, -1, -1, -1, -1, -1, -1, 0, 1, -1, 0],
    [0, 0, 1, 0, 0, -1, -1, -1, -1, 1, -1, 0, -1, -1, 0], // Bullish bias, but in Bearish OB (WAIT)
    [1, 1, 2, 1, 1, -1, -1, -1, -1, 0, -1, 0, -1, -1, 0], // Bearish bias, but in Bullish OB (WAIT)
    [0, 0, 1, 0, 0, -1, 1, -1, -1, -1, -1, 0, -1, -1, 0], // Bullish bias, but in Bearish FVG (WAIT)
    [1, 1, 2, 1, 1, -1, 0, -1, -1, -1, -1, 0, -1, -1, 0], // Bearish bias, but in Bullish FVG (WAIT)
    [0, 0, 1, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, -1, 0],
    [1, 1, 2, 1, 1, -1, -1, -1, -1, -1, -1, 0, 1, -1, 0],
    [0, 0, 1, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, -1, 0],
    [1, 1, 2, 1, 1, -1, -1, -1, -1, -1, -1, 0, 1, -1, 0],
    [0, 0, 1, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, -1, 0],
    [1, 1, 2, 1, 1, -1, -1, -1, -1, -1, -1, 0, 1, -1, 0],
    [0, 0, 1, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, -1, 0],
    [1, 1, 2, 1, 1, -1, -1, -1, -1, -1, -1, 0, 1, -1, 0],
    [0, 0, 1, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, -1, 0],
    [1, 1, 2, 1, 1, -1, -1, -1, -1, -1, -1, 0, 1, -1, 0],
    [0, 0, 1, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, -1, 0],
    [1, 1, 2, 1, 1, -1, -1, -1, -1, -1, -1, 0, 1, -1, 0],
    [0, 0, 1, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, -1, 0],
    [1, 1, 2, 1, 1, -1, -1, -1, -1, -1, -1, 0, 1, -1, 0],
    [0, 0, 1, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, -1, 0],
    [1, 1, 2, 1, 1, -1, -1, -1, -1, -1, -1, 0, 1, -1, 0],
];

function mostCommonLabel(y) {
 const counts = y.reduce((acc, label) => {
  acc[label] = (acc[label] || 0) + 1;
  return acc;
 }, {});
 let maxCount = -1;
 let maxLabel = null;
 // Iterate over keys (labels)
 for (const labelKey in counts) {
  const count = counts[labelKey];
  if (count > maxCount) {
   maxCount = count;
   // Ensure label is returned as a number if possible
   maxLabel = Number(labelKey);
  }
 }
 return maxLabel;
}

class RandomForestClassifier {
 constructor(max_depth = 4, min_samples_split = 2, criterion = "entropy", max_features = "sqrt") {
  this.max_depth = max_depth;
  this.min_samples_split = min_samples_split;
  this.criterion = criterion;
  this.max_features = max_features;
  this.trees = []; // Stores tuples: [tree_instance, feature_indices_used]
  this.nFeaturesToUse = 0;
 }

 fit(X, y) {
  this.trees = [];
  const nSamples = X.length;
  const nFeatures = X[0].length;
  
  // Determine the number of features to use per split
  if (this.max_features === "sqrt") {
   this.nFeaturesToUse = Math.floor(Math.sqrt(nFeatures));
  } else if (this.max_features === "log2") {
   this.nFeaturesToUse = Math.floor(Math.log2(nFeatures));
  } else if (typeof this.max_features === 'number') {
   this.nFeaturesToUse = this.max_features;
  } else {
   this.nFeaturesToUse = nFeatures;
  }
  
  // Ensure at least one feature is used
  if (this.nFeaturesToUse < 1) this.nFeaturesToUse = 1;
  
  let combinations = [], n = Array.from({ length: nFeatures }, (_, i) => i);
  getCombinations(n, this.nFeaturesToUse, i => combinations.push([...i]));
  
  console.log(`Training ${combinations.length} trees using ${this.nFeaturesToUse} random features per split...`);
  
  for (let i = 0; i < combinations.length; i++) {
   const tree = new DecisionTree({ 
    maxDepth: this.max_depth,
    minSamplesSplit: this.min_samples_split,
    criterion: this.criterion 
   });
   const featureIndices = combinations[i];
   let X_bootstrap_subset = X.map(row => featureIndices.map(featureIdx => row[featureIdx]));
   
   // Fit the tree
   tree.fit(X_bootstrap_subset, y);
   this.trees.push([tree, featureIndices]);
  }
  console.log("Random Forest training complete.");
 }

 predict(X, y) {
  if (this.trees.length === 0) throw new Error("Model not trained.");
  
  // Collect predictions from all trees
  const allPredictions = this.trees.map(([tree, featureIndices]) => {
   // Predict using only the features that the tree was trained on
   
   const X_subset = X.map(row => featureIndices.map(featureIdx => row[featureIdx]));
   
   return tree.predict(X_subset);
  });
  
  const nSamples = X.length;
  const finalPredictions = [];
  
  for (let sampleIdx = 0; sampleIdx < nSamples; sampleIdx++) {
   const samplePredictions = allPredictions.map(treePreds => treePreds[sampleIdx]);
   finalPredictions.push(mostCommonLabel(samplePredictions));
  }
  
  let returnVal = {
   predictions: finalPredictions,
   score: 0
  }
  
  if(y) {
   const correct = finalPredictions.filter((pred, i) => pred === y[i]).length;
   returnVal.score = correct / y.length;
  }
  
  return returnVal;
 }
}

// Model Training and Testing ---
function getCombinations(arr, size, cal) {
 function generate(c, start) {
  // If the current combination has the desired size, add it to the result
  if (c.length === size) {
   cal(c);
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

function runRandomForestTest() {
 console.log("--- Starting Random Forest Trading Model Test ---");
 
 // Separate features (X) and labels (y)
 const X = TRADING_DATA.map(row => row.slice(0, row.length -1)); // 14 features
 const y = TRADING_DATA.map(row => row[row.length -1]); // 1 label (TradeAction)
 
 // Shuffle X for X_test
 let numbers = Array.from({ length: X.length }, (_, i) => i);
 let shuffledIndices = numbers.sort(() => 0.5 - Math.random());
 shuffledIndices = shuffledIndices.slice(0, 10);
 
 const X_test = shuffledIndices.map(i => X[i]);
 const y_test = shuffledIndices.map(i => y[i]);
 
 // Initialize the Random Forest Classifier
 const rf = new RandomForestClassifier(
  3, // max_depth (Keep small to prevent overfitting)
  3, // min_samples_split
  "gini",
  4, // max_features
 );
 
 // Train the model
 rf.fit(X, y);
 
 setTimeout(() => {
  // Evaluate the model on the testing data
  
  const prediction = rf.predict(X_test, y_test);
  
  console.log("-----------------------------------------------");
  console.log(`Total Samples: ${X.length}`);
  console.log(`Training Samples: ${X.length}`);
  console.log(`Testing Samples: ${X_test.length}`);
  console.log(`Test Set Accuracy: ${(prediction.score * 100).toFixed(2)}%`);
  console.log("-----------------------------------------------");
  
  // Example Prediction on a specific test point
  console.log(`True Label:`, y_test);
  console.log(`Predicted:`, prediction.predictions);
 }, 1000);
}

runRandomForestTest;
