//+------------------------------------------------------------------+
//| Logistic Regression in MQL5 (no plotting, no pandas)             |
//+------------------------------------------------------------------+
#property strict

//--- Sigmoid function
double Sigmoid(double z) {
 return 1.0 / (1.0 + MathExp(-z));
}
//--- Cost function
double Cost(const double & h[],
 const double & y[], int n) {
 double sum = 0.0;
 for (int i = 0; i < n; i++) {
  // prevent log(0) by bounding
  double hi = MathMin(MathMax(h[i], 1e-12), 1.0 - 1e-12);
  sum += -y[i] * MathLog(hi) - (1.0 - y[i]) * MathLog(1.0 - hi);
 }
 return sum / n;
}
//--- Gradient calculation
void Gradient(const double & X[][],
 const double & h[], const double & y[],
   double & grad[], int rows, int cols) {
 ArrayInitialize(grad, 0.0);
 for (int j = 0; j < cols; j++) {
  double g = 0.0;
  for (int i = 0; i < rows; i++) g += X[i][j] * (h[i] - y[i]);
  grad[j] = g / rows;
 }
}
//--- Logistic Regression (training)
void LogisticRegression(const double & X[][],
 const double & y[], double & theta[], double alpha, int iters, int rows, int cols) {
 double h[];
 ArrayResize(h, rows);
 double grad[];
 ArrayResize(grad, cols);
 
 for (int iter = 0; iter < iters; iter++) {
  // hypothesis
  for (int i = 0; i < rows; i++) {
   double z = 0.0;
   for (int j = 0; j < cols; j++) z += X[i][j] * theta[j];
   h[i] = Sigmoid(z);
  }
  
  // optional debug
  // if(iter % (iters/10) == 0)
  //    PrintFormat("Iter %d | Cost %.6f", iter, Cost(h,y,rows));
  
  Gradient(X, h, y, grad, rows, cols);
  
  for (int j = 0; j < cols; j++)
   theta[j] -= alpha * grad[j];
 }
}
// Example Run
int OnInit() {
 // Example: small dataset (iris-like)
 // X = [ [1, length, width], ... ]  (bias + features)
 double X[5][3] = {
  { 1.0, 4.7, 3.2 },
  { 1.0, 5.1, 3.5 },
  { 1.0, 6.0, 3.9 },
  { 1.0, 5.5, 2.3 },
  { 1.0, 6.2, 2.8 }
 };
 double y[5] = { 0, 0, 1, 1, 1 }; // Binary labels
 
 int rows = ArrayRange(X, 0);
 int cols = ArrayRange(X, 1);
 
 double theta[];
 ArrayResize(theta, cols);
 ArrayInitialize(theta, 0.0);
 
 double alpha = 0.01;
 int iterations = 1000;
 
 // Initial cost
 double h[];
 ArrayResize(h, rows);
 for (int i = 0; i < rows; i++) {
  double z = 0.0;
  for (int j = 0; j < cols; j++) z += X[i][j] * theta[j];
  h[i] = Sigmoid(z);
 }
 PrintFormat("Initial Cost: %.6f", Cost(h, y, rows));
 
 // Train
 LogisticRegression(X, y, theta, alpha, iterations, rows, cols);
 
 // Final cost
 for (int i = 0; i < rows; i++) {
  double z = 0.0;
  for (int j = 0; j < cols; j++) z += X[i][j] * theta[j];
  h[i] = Sigmoid(z);
 }
 PrintFormat("Final Cost: %.6f", Cost(h, y, rows));
 
 // Print final theta
 string th_str = "";
 for (int j = 0; j < cols; j++) th_str += DoubleToString(theta[j], 6) + " ";
 Print("Final Theta: ", th_str);
 
 return INIT_SUCCEEDED;
}
//+------------------------------------------------------------------+
//| Example: run logistic regression on preprocessed MqlRates data   |
//+------------------------------------------------------------------+
int OnInit() {
 MqlRates rates[];
 int bars = CopyRates(_Symbol, PERIOD_H1, 0, 300, rates); // example
 if (bars <= 0)
 {
  Print("No rates data!");
  return INIT_FAILED;
 }
 
 // --- Preprocess into features + labels
 // Example: X = [1, close-open, high-low]
 int rows = bars;
 int cols = 3; // bias + 2 features
 double X[][3];
 ArrayResize(X, rows);
 double y[];
 ArrayResize(y, rows);
 
 for (int i = 0; i < rows; i++)
 {
  X[i][0] = 1.0; // bias
  X[i][1] = rates[i].close - rates[i].open; // feature 1
  X[i][2] = rates[i].high - rates[i].low; // feature 2
  
  // label: simple rule (example: 1 if bullish bar else 0)
  y[i] = (rates[i].close > rates[i].open) ? 1.0 : 0.0;
 }
 
 // --- Train
 double theta[];
 ArrayResize(theta, cols);
 ArrayInitialize(theta, 0.0);
 
 double alpha = 0.01;
 int iterations = 500;
 
 Print("Initial Cost: ", Cost(y, y, rows)); // note: h=y initially since theta=0
 
 LogisticRegression(X, y, theta, alpha, iterations, rows, cols);
 
 // --- Final Hypothesis
 double h[];
 ArrayResize(h, rows);
 for (int i = 0; i < rows; i++)
 {
  double z = 0.0;
  for (int j = 0; j < cols; j++)
   z += X[i][j] * theta[j];
  h[i] = Sigmoid(z);
 }
 
 PrintFormat("Final Cost: %.6f", Cost(h, y, rows));
 
 string th_str = "";
 for (int j = 0; j < cols; j++)
  th_str += DoubleToString(theta[j], 6) + " ";
 Print("Final Theta: ", th_str);
 
 return INIT_SUCCEEDED;
}
//--- Predict probability
double PredictProbability(const double & features[],
 const double & theta[], int cols) {
 double z = 0.0;
 for (int j = 0; j < cols; j++)
  z += features[j] * theta[j];
  
 return Sigmoid(z);
}

//--- Predict class (0 or 1, threshold 0.5)
int PredictClass(const double & features[],
 const double & theta[], int cols, double threshold = 0.5) {
  double z = 0.0;
  for (int j = 0; j < cols; j++) z += features[j] * theta[j];
 return (Sigmoid(z) >= threshold) ? 1 : 0;
}