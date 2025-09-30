//+------------------------------------------------------------------+ 
//|                        Candle Exporter.mq5                       | 
//|      Exports candle data to CSV for ML prediction in Python     | 
//+------------------------------------------------------------------+ 
#property script_show_inputs
input string FileName = "latest_candle.csv";
input int NumCandles = 20; // How many candles to export 
input ENUM_TIMEFRAMES Timeframe = PERIOD_CURRENT;

#include <Files\CSV.mqh>

void OnStart() {
 string fullPath = TerminalInfoString(TERMINAL_COMMONDATA_PATH) + "\MQL5\Files\" + FileName; ResetLastError(); 
 
 string buffer = "open,high,low,close,volume,bias!!&& \n";
 
 MqlRates rates[];
 int copied = CopyRates(_Symbol, Timeframe, 0, NumCandles, rates);
 
 if (copied <= 0) {
  Print("CopyRates failed: ", GetLastError());
  FileClose(file);
  return;
 }
 
 for (int i = NumCandles - 1; i >= 0; i--) {
  buffer += StringFormat("%.*f,%.*f,%.*f,%.*f,%d,%s\n",
   _Digits, rates[i].open,
   _Digits, rates[i].high,
   _Digits, rates[i].low,
   _Digits, rates[i].close,
   (rates[i].close >= rates[i].open ? 1 : 0)
  );
 }
 
 int file = FileOpen(FileName, FILE_WRITE | FILE_ANSI);
 if (file != INVALID_HANDLE) {
  FileWriteString(file, buffer);
  FileClose(file);
  Print("[Init] Exported ", NumCandles, " candles to ", fullPath);
 } else {
  Print("Failed to open file: ", GetLastError());
 }
}


#property strict
input int LookbackCandles = 5;
input int TrainingSetSize = 1000;

datetime lastExportTime = 0;

// Training set generator
void GenerateTrainingCSV() {
 MqlRates rates[];
 if (CopyRates(_Symbol, PERIOD_CURRENT, 1, TrainingSetSize + 4, rates) < TrainingSetSize + 4)
  return;
 
 int handle = FileOpen("training_data.csv", FILE_WRITE | FILE_CSV | FILE_ANSI);
 if (handle == INVALID_HANDLE) {
  Print("Error opening training file: ", GetLastError());
  return;
 }
 
 FileWrite(handle, "open", "high", "low", "close", "bias");
 
 for (int i = TrainingSetSize - 1; i >= 0; i--) {
  FileWrite(handle,
   rates[i].open,
   rates[i].high,
   rates[i].low,
   rates[i].close,
   (rates[i].close < rates[i - 1].close) ? 0 : 1
  );
 }
 
 FileClose(handle);
 Print("Training set exported.");
}

// Live update for current candle
void ExportLatestCandles() {
 MqlRates rates[];
 if (CopyRates(_Symbol, PERIOD_CURRENT, 1, LookbackCandles + 4, rates) < LookbackCandles + 4)
  return;
 
 int handle = FileOpen("latest_candle.csv", FILE_WRITE | FILE_CSV | FILE_ANSI);
 if (handle == INVALID_HANDLE) {
  Print("Error opening live file: ", GetLastError());
  return;
 }
 
 FileWrite(handle, "open", "high", "low", "close", "bias");
 
 for (int i = LookbackCandles - 1; i >= 0; i--) {
  FileWrite(handle,
   rates[i].open,
   rates[i].high,
   rates[i].low,
   rates[i].close,
   (rates[i].close < rates[i - 1].close) ? 0 : 1
  );
 }
 
 FileClose(handle);
 lastExportTime = rates[0].time;
}

int OnInit() {
 GenerateTrainingCSV(); // Once on init
 return INIT_SUCCEEDED;
}

void OnTick() {
 datetime newTime = iTime(_Symbol, PERIOD_CURRENT, 1);
 if (newTime != lastExportTime)
  ExportLatestCandles();
}