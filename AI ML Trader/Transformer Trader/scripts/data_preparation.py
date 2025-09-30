import pandas as pd
import numpy as np
import torch
import pandas_ta as ta
import pickle
from trading_api.mt5_connector import MT5Connector
from sklearn.preprocessing import StandardScaler

features = ['high', 'low', 'close', 'log_ret', 'high_low', 'high_open', 'open_low', 'obv', 'ema', 'cci', 'hour']
# Helper function to get data from MT5
def get_mt5_data(mt5, symbol, timeframe, bars):
 rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
 return pd.DataFrame(rates)

def create_features(df, scaler):
 """
 Creates and normalizes a rich set of features from OHLCV data.
 """
 # 1. Price and Volatility Features
 df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
 df['high_low'] = df['high'] - df['low']
 df['high_open'] = df['high'] - df['open']
 df['open_low'] = df['open'] - df['low']
 
 # 2. Technical Indicator Features
 df['obv'] = ta.obv(close=df['close'], volume=df['volume'])
 df['ema'] = ta.ema(close=df['close'], length=50)
 df['cci'] = ta.CCI(df['high'], df['low'], df['close'], timeperiod=30)
 
 # 3. Time-based Features
 df['hour'] = df.index.hour
 
 # Find the first index where all columns are valid
 first_valid_index = data.dropna().index[0]
 
 # Slice the DataFrame from this index to maintain alignment
 clean_data = data.loc[first_valid_index:].copy()
 
 # 5. Normalize the continuous features
 featuresN = ['high', 'low', 'close', 'log_ret', 'high_low', 
  'high_open', 'open_low', 'obv', 'ema', 'cci']
 scalerN = None
 if scaler: 
  df[featuresN] = scaler.transform(df[featuresN])
 else:
  scalerN = StandardScaler()
  df[featuresN] = scalerN.fit_transform(df[featuresN])
 
 return df, scalerN

def prepare_data():
 """
 Collects, aligns, and normalizes multi-timeframe OHLCV data.
 """
 symbol = "EURUSD"
 connector = MT5Connector(self, login, password, server, symbol, scaler=False)
 
 mt5 = connector.mt5
 
 # Collect data for two different timeframes
 df_4h = get_mt5_data(mt5, symbol, mt5.TIMEFRAME_H4, 5000)
 df_15m = get_mt5_data(mt5, symbol, mt5.TIMEFRAME_M15, 20000)
 
 # Convert to datetime and set as index
 df_4h['time'] = pd.to_datetime(df_4h['time'], unit='s')
 df_15m['time'] = pd.to_datetime(df_15m['time'], unit='s')
 df_4h.set_index('time', inplace=True)
 df_15m.set_index('time', inplace=True)
 

 # Align 15M to 4H
 # Todo: 
 
 # Feature Engineering and Normalization
 df_4h, scaler4h = create_features(df_4h)
 df_15m, scaler15m = create_features(df_15m)
 
 # Labeling for HTF model
 # Labels based on future 3 bars (up=1, down=0, side=2)
 df_4h['label'] = 0 # Placeholder for labeling logic
 # Implement labeling based on 'future bars' logic here
 
 # Labeling for LTF Entry model
 # Labels based on entry/drawdown criteria
 df_15m['label'] = 0 # Placeholder for labeling logic
 # Implement labeling based on 'target return' logic here

 # Convert to PyTorch tensors and save
 htf_data = torch.tensor(df_4h[features].values, dtype=torch.float32)
 htf_labels = torch.tensor(df_4h['label'].values, dtype=torch.long)
 torch.save((htf_data, htf_labels), 'data/htf_data.pt')

 ltf_data = torch.tensor(df_15m[features].values, dtype=torch.float32)
 ltf_labels = torch.tensor(df_15m['label'].values, dtype=torch.long)
 torch.save((ltf_data, ltf_labels), 'data/ltf_data.pt')
 
 with open("data/scaler4h.pkl", "wb") as f:
  pickle.dump(scaler4h, f)
  print(f"4H scaler saved to {f}")
 
 with open("data/scaler15m.pkl", "wb") as f:
  pickle.dump(scaler15m, f)
  print(f"15M scaler saved to {f}")
 
 mt5.shutdown()

if __name__ == '__main__':
 prepare_data()
 print("Data preparation complete.")




def create_features(df):
    features = ['open', 'high', 'low', 'close', 'tick_volume']
    
    # Example TA-Lib indicators
    df['ema'] = talib.EMA(df['close'], timeperiod=30)
    df['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=30)
    
    # Add other features here
    # df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    
    feature_columns = ['ema', 'cci'] # Add other indicator names here
    
    # Normalize features
    scaler = MinMaxScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    
    return df, scaler, feature_columns

# --- Main Logic ---
def prepare_data(login, password, server, symbol):
    """
    Collects, aligns, and normalizes multi-timeframe OHLCV data.
    """
    
    # Initialize connector
    connector = MT5Connector(login=login, password=password, server=server, symbol=symbol, scaler=False)
    mt5_instance = connector.mt5
    
    # Collect data for two different timeframes
    df_4h = connector.get_mt5_data(mt5_instance.TIMEFRAME_H4, 5000)
    df_15m = connector.get_mt5_data(mt5_instance.TIMEFRAME_M15, 20000)
    
    # Convert to datetime and set as index
    df_4h['time'] = pd.to_datetime(df_4h['time'], unit='s')
    df_15m['time'] = pd.to_datetime(df_15m['time'], unit='s')
    df_4h.set_index('time', inplace=True)
    df_15m.set_index('time', inplace=True)
    
    # --- Feature Engineering and Alignment ---
    df_4h, scaler4h, features_4h = create_features(df_4h)
    df_15m, scaler15m, features_15m = create_features(df_15m)

    # Shift higher timeframe features to prevent look-ahead bias
    df_4h_shifted = df_4h.shift(1)

    # Create alignment key in 15M dataframe
    df_15m['4h_open_time'] = df_15m.index.floor('4H')

    # Merge the shifted 4H data into the 15M dataframe
    df_15m = pd.merge(df_15m, df_4h_shifted[features_4h],
                      left_on='4h_open_time', right_index=True, how='left',
                      suffixes=('', '_4h'))

    # Drop any remaining NaNs, which also handles misaligned data at the beginning
    df_15m.dropna(inplace=True)
    
    # --- Labeling ---
    
    # Labeling for HTF model (as per your todo)
    df_4h['label'] = 0
    # Your labeling logic goes here, e.g., based on future price movement
    
    # Labeling for LTF model (as per your todo)
    # The target needs to be calculated on the CLEANED and ALIGNED dataframe
    df_15m['label'] = 0
    # Your labeling logic goes here, e.g., a shift to predict next bar
    df_15m.dropna(inplace=True) # Ensure no NaNs from labeling

    # --- Data Saving ---
    
    # Align features for both models after final data cleaning
    htf_data = torch.tensor(df_4h[features_4h].values, dtype=torch.float32)
    htf_labels = torch.tensor(df_4h['label'].values, dtype=torch.long)
    torch.save((htf_data, htf_labels), 'data/htf_data.pt')
    
    # Adjust feature list for the LTF model to include the new 4h features
    ltf_features = features_15m + [f'{col}_4h' for col in features_4h]
    ltf_data = torch.tensor(df_15m[ltf_features].values, dtype=torch.float32)
    ltf_labels = torch.tensor(df_15m['label'].values, dtype=torch.long)
    torch.save((ltf_data, ltf_labels), 'data/ltf_data.pt')
    
    # Save scalers
    with open("data/scaler4h.pkl", "wb") as f:
        pickle.dump(scaler4h, f)
        print("4H scaler saved.")
    
    with open("data/scaler15m.pkl", "wb") as f:
        pickle.dump(scaler15m, f)
        print("15M scaler saved.")
    
    mt5_instance.shutdown()






def prepare_data_from_mt5():
    """
    Collects, aligns, and normalizes multi-timeframe OHLCV data directly from MT5.
    """
    if not mt5.initialize():
        print("MetaTrader5 initialization failed")
        return

    # Helper function to get data from MT5
    def get_mt5_data(symbol, timeframe, bars):
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
        return pd.DataFrame(rates)

    # Collect data for two different timeframes
    df_4h = get_mt5_data('EURUSD', mt5.TIMEFRAME_H4, 5000)
    df_15m = get_mt5_data('EURUSD', mt5.TIMEFRAME_M15, 20000)

    df_4h['time'] = pd.to_datetime(df_4h['time'], unit='s')
    df_15m['time'] = pd.to_datetime(df_15m['time'], unit='s')
    df_4h.set_index('time', inplace=True)
    df_15m.set_index('time', inplace=True)

    # Create features for each DataFrame
    df_4h_processed, htf_scaler = create_features(df_4h)
    df_15m_processed, ltf_scaler = create_features(df_15m)

    # Get the final list of features and their count for the HTF model
    htf_features_list = df_4h_processed.columns.tolist()
    htf_feature_dim = len(htf_features_list)
    print(f"HTF features list: {htf_features_list}")
    print(f"HTF feature dimension (feature_dim): {htf_feature_dim}")

    # Get the final list of features and their count for the LTF model
    ltf_features_list = df_15m_processed.columns.tolist()
    ltf_feature_dim = len(ltf_features_list)
    print(f"LTF features list: {ltf_features_list}")
    print(f"LTF feature dimension (feature_dim): {ltf_feature_dim}")

    # Now, save the processed data and the final feature lists/dimensions
    torch.save(
        {
            'htf_data': torch.tensor(df_4h_processed.values, dtype=torch.float32),
            'htf_labels': torch.tensor(df_4h_processed['ema_label'].values, dtype=torch.long),
            'htf_feature_dim': htf_feature_dim,
            'htf_features_list': htf_features_list
        }, 
        'data/processed/htf_data.pt'
    )
    
    torch.save(
        {
            'ltf_data': torch.tensor(df_15m_processed.values, dtype=torch.float32),
            'ltf_labels': torch.tensor(df_15m_processed['ema_label'].values, dtype=torch.long),
            'ltf_feature_dim': ltf_feature_dim,
            'ltf_features_list': ltf_features_list
        },
        'data/processed/ltf_data.pt'
    )

if __name__ == '__main__':
    prepare_data_from_mt5()
    print("Data preparation complete.")
