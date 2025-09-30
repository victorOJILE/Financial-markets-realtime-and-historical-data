//+------------------------------------------------------------------+
//|                                     Trading_TrendClassifier.mq5  |
//|                                      Copyright 2025, Gemini      |
//|                                                                  |
//| This EA uses a logistic regression model with Ridge regularization|
//| to classify market trends based on OHLC, Volume, and Volatility. |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, Gemini"
#property version "1.00"
#property strict

// --- Input parameters
input int InpTrainingPeriod = 200; // Number of bars to train the model on
input double InpLambda = 0.01; // Ridge regularization penalty parameter
input int InpVolatilityPeriod = 14; // Period for Average True Range
input double InpTrendThreshold = 0.005; // Percentage change to define a "trend"

// --- Global variables
double openPrices[];
double highPrices[];
double lowPrices[];
double closePrices[];
double volume[];
double atrValues[]; // Volatility metric

matrix X; // Matrix of independent variables
vector Y; // Vector of dependent (trend) variables
vector Beta; // Vector of regression coefficients

int atrHandle;

int OnInit() {
 ArraySetAsSeries(openPrices, true);
 ArraySetAsSeries(highPrices, true);
 ArraySetAsSeries(lowPrices, true);
 ArraySetAsSeries(closePrices, true);
 ArraySetAsSeries(volume, true);
 ArraySetAsSeries(atrValues, true);
 
 atrHandle = iATR(_Symbol, _Period, InpVolatilityPeriod);
 if (atrHandle == INVALID_HANDLE) {
  Print("Failed to get ATR handle.");
  return INIT_FAILED;
 }

 // --- Initialize matrices and vectors
 // 5 features: Open, High, Low, Volume, ATR
 // + 1 for the intercept
 X.Init(InpTrainingPeriod, 6);
 Y.Init(InpTrainingPeriod);
 Beta.Init(6);
 
 // --- Initial training of the model
 TrainModel();
 
 return (INIT_SUCCEEDED);
}

void OnDeinit(const int reason) {}

void OnTick() {
 // --- Check for new bar
 static datetime lastBarTime = 0;
 datetime currentBarTime = iTime(_Symbol, _Period, 0);
 if (currentBarTime == lastBarTime) return false;
 lastBarTime = currentBarTime;
 
 // --- Retrain the model on new data
 TrainModel();
 
 // --- Get the trend probability
 double trendProbability = PredictTrend();
 
 // --- Make a trading decision based on the probability
 int decision = MakeDecision(trendProbability);
 
 // --- Execute the trade
 ExecuteTrade(decision);
}

//+------------------------------------------------------------------+
//| Trains the logistic regression model                             |
//| This simplified version uses a linear approximation for training |
//| for demonstrative purposes, as a true logistic regression training|
//| process (e.g., gradient descent) is far too complex for this example.|
//+------------------------------------------------------------------+
void TrainModel() {
 // --- Copy market data
 if (CopyOpen(_Symbol, _Period, 1, InpTrainingPeriod, openPrices) != InpTrainingPeriod ||
  CopyHigh(_Symbol, _Period, 1, InpTrainingPeriod, highPrices) != InpTrainingPeriod ||
  CopyLow(_Symbol, _Period, 1, InpTrainingPeriod, lowPrices) != InpTrainingPeriod ||
  CopyClose(_Symbol, _Period, 1, InpTrainingPeriod, closePrices) != InpTrainingPeriod ||
  CopyVolume(_Symbol, _Period, 1, InpTrainingPeriod, volume) != InpTrainingPeriod) {
  Print("Failed to copy historical data.");
  return;
 }
 
 // --- Calculate ATR as a volatility feature
 if (CopyBuffer(atrHandle, 0, 1, InpTrainingPeriod, atrValues) != InpTrainingPeriod) {
  Print("Failed to copy ATR values.");
  return;
 }
 
 // --- Populate the matrices and vectors
 for (int i = 0; i < InpTrainingPeriod; i++) {
  // --- Create the 'Y' dependent variable for classification
  // A trend is defined by a significant change in price over the next bar.
  // We're classifying based on the change from bar i+1 to bar i.
  if (i > 0) {
   double priceChange = (closePrices[i - 1] - closePrices[i]) / closePrices[i];
   if (priceChange > InpTrendThreshold) {
    Y.Set(i - 1, 1.0); // Upward trend
   } else if (priceChange < -InpTrendThreshold) {
    Y.Set(i - 1, -1.0); // Downward trend
   } else {
    Y.Set(i - 1, 0.0); // Sideways
   }
  }
  
  // --- Populate the X matrix with features for bar 'i'
  X.Set(i, 0, 1.0); // Intercept
  X.Set(i, 1, openPrices[i]);
  X.Set(i, 2, highPrices[i]);
  X.Set(i, 3, lowPrices[i]);
  X.Set(i, 4, volume[i]);
  X.Set(i, 5, atrValues[i]); // Volatility metric
 }
 
 // --- Adjust matrices to match the size of the dependent variable
 matrix tempX;
 tempX.Init(InpTrainingPeriod - 1, 6);
 for (int i = 0; i < InpTrainingPeriod - 1; i++) {
  tempX.Row(i, X.Row(i + 1));
 }
 X = tempX;
 Y.Resize(InpTrainingPeriod - 1);
 
 // --- Implement Ridge Regularization for the matrix inversion
 matrix XT = X.Transpose();
 matrix identity;
 identity.Init(6, 6);
 identity.Identity();
 matrix XT_X = XT.MatMul(X);
 matrix regularizedXT_X = XT_X + InpLambda * identity;
 
 matrix XT_X_Inv;
 if (regularizedXT_X.Inverse(XT_X_Inv)) {
  vector XT_Y = XT.MatMul(Y);
  Beta = XT_X_Inv.MatMul(XT_Y);
  
  Print("Model retrained successfully. Beta coefficients: ", Beta.ToString());
 } else {
  Print("Matrix inversion failed. Adjust InpLambda or check for errors.");
 }
}

//+------------------------------------------------------------------+
//| Predicts the trend probability for the next bar                  |
//+------------------------------------------------------------------+
double PredictTrend() {
 // --- Get the most recent features
 double currentOpen = iOpen(_Symbol, _Period, 0);
 double currentHigh = iHigh(_Symbol, _Period, 0);
 double currentLow = iLow(_Symbol, _Period, 0);
 double currentVolume = iVolume(_Symbol, _Period, 0);
 double currentATR;
 
 CopyBuffer(atrHandle, 0, 1, InpTrainingPeriod, currentATR);
 
 // --- Create a vector of independent variables for the prediction
 vector X_new;
 X_new.Init(6);
 X_new.Set(0, 1.0); // Intercept
 X_new.Set(1, currentOpen);
 X_new.Set(2, currentHigh);
 X_new.Set(3, currentLow);
 X_new.Set(4, currentVolume);
 X_new.Set(5, currentATR);
 
 // --- Calculate the raw output
 vector rawOutput = X_new.MatMul(Beta);
 
 // --- Apply the sigmoid function to get a probability
 // For simplicity, we are modeling a single logistic regression.
 // For a multi-class problem (up/down/sideways), you would need to use
 // more advanced techniques like one-vs-rest or softmax.
 double probability = 1.0 / (1.0 + MathExp(-rawOutput.At(0)));
 
 return probability;
}

//+------------------------------------------------------------------+
//| Make the trading decision (0=HOLD, 1=BUY, 2=SELL)                |
//+------------------------------------------------------------------+
int MakeDecision(double trendProbability) {
 // A simplified decision making process. A probability close to 1
 // indicates a strong "up" trend, while close to 0 indicates "down".
 // A probability around 0.5 suggests a sideways market.
 if (trendProbability > 0.6) {
  return 1; // BUY (strong likelihood of upward trend)
 } else if (trendProbability < 0.4) {
  return 2; // SELL (strong likelihood of downward trend)
 }
 
 return 0; // HOLD
}