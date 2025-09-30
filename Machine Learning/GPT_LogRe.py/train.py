import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

INPUT_PATH = Path("training_data.csv") # MQL5 will write this

def train() :
    df = pd.read_csv(INPUT_PATH)
    
    X = df[['open', 'high', 'low', 'close', 'bias']]
    y = df['bias']
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000))
    ])
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Save model
    model_path = "train.pkl"
    joblib.dump(pipeline, model_path)
    
    # Save scaler
    # ...
    
if INPUT_PATH.exists():
    train()
else : 
    print(f"Training data not available at {INPUT_PATH}")
