// Creates an array/matrix filled with ones.
const ones = (rows, cols) => Array(rows).fill(0).map(() => Array(cols).fill(1));

// Creates an array/matrix filled with zeros.
const zeros = (rows, cols) => Array(rows).fill(0).map(() => Array(cols).fill(0));

// Calculates the transpose of a matrix.
const transpose = (A) => {
 const rows = A.length;
 const cols = A[0].length;
 const AT = zeros(cols, rows);
 for (let i = 0; i < rows; i++) {
  for (let j = 0; j < cols; j++) AT[j][i] = A[i][j];
 }
 return AT;
};

// Multiplies two matrices (Dot Product).
const multiply = (A, B) => {
 const rowsA = A.length;
 const colsA = A[0].length;
 const rowsB = B.length;
 const colsB = B[0].length;
 
 if (colsA !== rowsB) {
  throw new Error("Matrix dimensions are incompatible for multiplication.");
 }
 
 const C = zeros(rowsA, colsB);
 
 for (let i = 0; i < rowsA; i++) {
  for (let j = 0; j < colsB; j++) {
   let sum = 0;
   for (let k = 0; k < colsA; k++) sum += A[i][k] * B[k][j];
   C[i][j] = sum;
  }
 }
 return C;
};

// Subtracts two matrices (element-wise).
const subtract = (A, B) => {
 const C = [];
 for (let i = 0; i < A.length; i++) {
  C[i] = [];
  for (let j = 0; j < A[0].length; j++) C[i][j] = A[i][j] - B[i][j];
 }
 return C;
};

// Performs element-wise power on a matrix (column vector of features).
const dotPower = (A, p) => A.map(row => row.map(val => Math.pow(val, p)));

// Calculates the mean of each column (feature) in a matrix.
const mean = (A) => {
 const rows = A.length;
 const cols = A[0].length;
 const means = Array(cols).fill(0);
 
 for (let j = 0; j < cols; j++) {
  let sum = 0;
  for (let i = 0; i < rows; i++) sum += A[i][j];
  means[j] = sum / rows;
 }
 return means;
};

// Calculates the standard deviation of each column (feature) in a matrix.
const stdDev = (A, feature_means) => {
 const rows = A.length;
 const cols = A[0].length;
 const stds = Array(cols).fill(0);
 
 for (let j = 0; j < cols; j++) {
  let sumOfSquares = 0;
  for (let i = 0; i < rows; i++) sumOfSquares += Math.pow(A[i][j] - feature_means[j], 2);
  // Using N for sample size (common for machine learning/population std)
  stds[j] = Math.sqrt(sumOfSquares / rows);
 }
 return stds;
};

class PolynomialRegression {
 constructor(degree, learning_rate, iterations) {
  this.degree = degree;
  this.learning_rate = learning_rate;
  this.iterations = iterations;
  this.W = null; // Weights (column vector)
  this.m = 0; // Number of training examples
  this.X_norm_mean = null; // Mean array for normalization
  this.X_norm_std = null; // Std dev array for normalization
 }

 transform(X) {
  const m = X.length;
  let X_transform = ones(m, 1);
  for (let j = 1; j <= this.degree; j++) {
   const x_pow = dotPower(X, j);
   const newCols = X_transform[0].length + 1;
   let temp = Array(m).fill(0).map((_, i) => [...X_transform[i], x_pow[i][0]]);
   X_transform = temp;
  }
  return X_transform;
 }
 
 normalize(X, is_fitting = false) {
  const rows = X.length;
  const degree = X[0].length - 1;
  const X_features = X.map(row => row.slice(1));
  
  if (is_fitting) {
   this.X_norm_mean = mean(X_features);
   this.X_norm_std = stdDev(X_features, this.X_norm_mean);
  }

  const X_normalized_features = Array(rows).fill(0).map((_, i) => Array(degree).fill(0));
  
  for (let i = 0; i < rows; i++) {
   for (let j = 0; j < degree; j++) {
    const std = this.X_norm_std[j];
    const meanVal = this.X_norm_mean[j];
    
    if (std === 0) {
     // Avoid division by zero: keep constant features as is
     X_normalized_features[i][j] = X_features[i][j];
    } else {
     X_normalized_features[i][j] = (X_features[i][j] - meanVal) / std;
    }
   }
  }
  const X_normalize = X.map((row, i) => [row[0], ...X_normalized_features[i]]);
  return X_normalize;
 }

 fit(X_data, Y_data) {
  const X = X_data.map(x => [x]);
  const Y = Y_data.map(y => [y]);
  
  this.m = X.length;
  
  const X_transform = this.transform(X);
  const X_normalize = this.normalize(X_transform, true);

  this.W = zeros(this.degree + 1, 1);

  for (let i = 0; i < this.iterations; i++) {
   const h = multiply(X_normalize, this.W);
   const error = subtract(h, Y);
   const X_normalize_T = transpose(X_normalize);
   const gradient_unscaled = multiply(X_normalize_T, error);
   for (let k = 0; k <= this.degree; k++) this.W[k][0] = this.W[k][0] - this.learning_rate * (1 / this.m) * gradient_unscaled[k][0];
  }
  
  return this;
 }
 
 // Predicts the target values for a given input feature set.
 predict(X_data) {
  if (!this.W) {
   throw new Error("Model not trained. Call fit() first.");
  }

  const X = X_data.map(x => [x]);
  const X_transform = this.transform(X);
  const X_normalize = this.normalize(X_transform, false);
  const h = multiply(X_normalize, this.W);
  
  return h.flat();
 }
}

// Sample Data: Trying to fit a curve (e.g., y = x^2 + noise)

const Y_train = [242.068, 296.525, 424.113, 274.668, 335.974, 578.621, 394.441, -1960.06]; // Values close to x^2
const X_train = Array.from({ length: 6 }, (_, i) => i);
X_train.shift();
X_train.reverse();
// Create and train the model
const degree = 2; // We expect a quadratic fit
const learning_rate = 0.01;
const iterations = 1000;
/*
// Make predictions
const X_test = [0]; //Array.from({ length: 7 }, (_, i) => i);
for(let i = 0; i < Y_train.length; i++) {
 let train = Y_train.slice(i, i +5);
 if(train.length < 5) break;
 
 const model = new PolynomialRegression(2, 0.01, 10000);
 model.fit(X_train, train);
 
 const predictions = model.predict(X_test);
 
 console.log('Train Y', train);
 console.log(`Prediction ${predictions}`);
}
*/
// Expected output for X=6 should be around 36