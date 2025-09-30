import time
import pandas as pd
import joblib
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# === Paths ===
MODEL_PATH = Path("train.pkl")
CSV_PATH = Path("latest_candle.csv")
OUTPUT_PATH = Path("bias_prediction.txt")

# === Load model ===
model = joblib.load(MODEL_PATH)
print("[Model loaded]")

# Load scaler
# ...

# === Prediction Function ===
def predict_bias(csv_path):
    try:
        df = pd.read_csv(CSV_PATH)
        X = df[['open', 'high', 'low', 'close', 'bias']]
        
        # Use scaler from training
        # ...
        
        # Predict latest row
        prediction = model.predict(X)[-1]
        
        with open(OUTPUT_PATH, 'w') as f:
            f.write(prediction)

        print(f"[Prediction updated] Bias: {prediction}")
    except Exception as e:
        print(f"[Error] {e}")

# === File watcher ===
class CSVChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith(CSV_PATH.name):
            predict_bias(CSV_PATH)

# === Monitor function ===
def watch_csv_file():
    event_handler = CSVChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, str(CSV_PATH.parent), recursive=False)
    observer.start()
    print(f"[Watching for updates to {CSV_PATH}...]")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

# === Start ===
if __name__ == "__main__":
    if not CSV_PATH.exists():
        print(f"Waiting for {CSV_PATH} to appear...")
        while not CSV_PATH.exists():
            time.sleep(1)

    predict_bias(CSV_PATH)  # Run once on startup
    watch_csv_file()
