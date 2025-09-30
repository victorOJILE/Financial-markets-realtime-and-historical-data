import torch, pickle, os

def save_model(model, path):
 os.makedirs(os.path.dirname(path), exist_ok=True)
 torch.save(model.state_dict(), path)
 print(f"✅ Model saved at {path}")

def load_model(model, path):
 model.load_state_dict(torch.load(path))
 model.eval()
 return model

def save_scaler(scaler, path):
 os.makedirs(os.path.dirname(path), exist_ok=True)
 with open(path, "wb") as f:
  pickle.dump(scaler, f)
  print(f"✅ Scaler saved at {path}")

def load_scaler(path):
 with open(path, "rb") as f:
  return pickle.load(f)
