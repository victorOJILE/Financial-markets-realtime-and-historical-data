import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Assuming these modules exist in your environment
from data.mt5_data import fetch_mt5_data, preprocess
from utils.save_utils import save_model, save_scaler

## ------------------ Compatible Classes ------------------ ##

# The TimeseriesDataset class that works with the first LSTMEntry model.
# It creates samples with an input sequence (X) and a target (y).
class TimeseriesDataset(Dataset):
    def __init__(self, arr, input_len=100, forecast_horizon=1):
        self.arr = arr
        self.input_len = input_len
        self.forecast_horizon = forecast_horizon
        self.X, self.y = self._create_windows()

    def _create_windows(self):
        X, y = [], []
        for i in range(len(self.arr) - self.input_len - self.forecast_horizon):
            X.append(self.arr[i:i + self.input_len])
            y.append(self.arr[i + self.input_len + self.forecast_horizon - 1][3])  # close price target
        return torch.tensor(X), torch.tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# The LSTMEntry class that works with the first TimeseriesDataset class.
# It takes a single input sequence `x`.
class LSTMEntry(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, context=None):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        if context is not None:
            out = torch.cat([out, context], dim=-1)
        return self.fc(out)


## ------------------ Main Script ------------------ ##

# Load and preprocess data
df = fetch_mt5_data(timeframe="M15")
arr = preprocess(df)
# Scale data
scaler = MinMaxScaler()
arr_scaled = scaler.fit_transform(arr)

# Dataset configuration
input_len = 100
forecast_horizon = 5

# Create a compatible dataset instance and DataLoader
dataset = TimeseriesDataset(arr_scaled, input_len=input_len, forecast_horizon=forecast_horizon)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model configuration, optimizer, and criterion
input_dim = arr_scaled.shape[1]
output_dim = 3  # For a 3-class classification: e.g., Buy, Sell, Hold
model = LSTMEntry(input_dim=input_dim, output_dim=output_dim)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# The training loop that correctly uses the dataset and model
print("Starting training...")
for epoch in range(10):
    for X, y in loader:
        # Get logits from the model
        logits = model(X)
        # Convert `y` (close price) to a classification label.
        # This is a proxy for Buy/Sell/Hold based on price movement.
        # We need to map y values to classes (0, 1, 2).
        # For simplicity, let's classify based on the price change from the previous period.
        # This is an example, and a real-world scenario would require more complex labeling.
        # Let's assume a simple rule: if price goes up significantly -> Buy, down -> Sell, stable -> Hold.
        # We'll use a simplified classification proxy: 0 for price drop, 1 for small change, 2 for price increase
        
        # A simple classification rule (requires a sorted y for proper comparison)
        y_np = y.numpy()
        labels = np.zeros_like(y_np)
        labels[y_np > np.percentile(y_np, 66)] = 2  # top 33% = class 2 (Buy)
        labels[y_np < np.percentile(y_np, 33)] = 0  # bottom 33% = class 0 (Sell)
        labels[(labels != 0) & (labels != 2)] = 1 # remaining 33% = class 1 (Hold)
        labels = torch.tensor(labels, dtype=torch.long)

        # Make sure labels tensor has the correct shape for CrossEntropyLoss
        loss = criterion(logits, labels)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}, Loss {loss.item():.4f}")
print("Training complete.")

# Save artifacts
save_model(model, "checkpoints/entry_lstm.pth")
save_scaler(scaler, "checkpoints/entry_scaler.pkl")
