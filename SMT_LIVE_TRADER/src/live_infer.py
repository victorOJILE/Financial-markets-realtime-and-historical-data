import torch
import time
import MetaTrader5 as mt5
import csv
from datetime import datetime
from model import SMTTransformer, LABELS
from mt5_utils import prepare_live_tensor, execute_trade

PAIR1 = "EURUSD"
PAIR2 = "GBPUSD"
TIMEFRAMES = ["M5", "M15", "H1"]
MODEL_PATH = "../models/smt_model.pt"

def log_trade(signal):
    with open('../data/ohlc_logs.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().isoformat(), LABELS[signal]])

def run_inference(model, tensor_input):
    model.eval()
    with torch.no_grad():
        logits = model(tensor_input.unsqueeze(0))  # Add batch dimension
        pred = torch.argmax(logits, dim=1).item()
        return pred, logits

if __name__ == "__main__":
    mt5.initialize()
    model = SMTTransformer()
    model.load_state_dict(torch.load(MODEL_PATH))
    print("Starting... Press Ctrl C to terminate")

    while True:
        print("Fetching live OHLC...")
        input_tensor = prepare_live_tensor(PAIR1, PAIR2, TIMEFRAMES)
        signal, _ = run_inference(model, input_tensor)
        print(f"Live Signal: {LABELS[signal]}")
        execute_trade(PAIR1, signal)
        execute_trade(PAIR2, signal)
        log_trade(signal)
        time.sleep(300) # 5 minutes
