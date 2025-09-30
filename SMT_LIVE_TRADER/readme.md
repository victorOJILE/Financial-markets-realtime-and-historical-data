# SMT Divergence Live Trading Bot

This project uses a Transformer-based neural network to detect SMT divergence between two forex pairs (like EURUSD and GBPUSD) in real-time. When a valid divergence signal appears, the bot places a live trade using MetaTrader 5.

---

## Install all with:
pip install -r requirements.txt

To run your bot:
cd smt_live_trader/src
python live_infer.py

Or double click it (making sure python is installed and configured on your system)

## To run training:
cd smt_live_trader/src
python train.py

### After training
The file `models/smt_model.pt` will be ready to use in your live trading bot (live_infer.py)
