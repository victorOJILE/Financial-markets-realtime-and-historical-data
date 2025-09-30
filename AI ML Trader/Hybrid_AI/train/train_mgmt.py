import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from data.dataset import TimeseriesDataset
from data.mt5_data import fetch_mt5_data, preprocess
from utils.save_utils import save_model, save_scaler

class LSTMManagement(nn.Module):
    def __init__(self, mng_feat_dim, state_dim, hidden_dim=128, n_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(mng_feat_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout)
        self.state_proj = nn.Linear(state_dim, hidden_dim)
        self.head = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2))  # Hold/Close

    def forward(self, mng_seq, trade_state):
        out, _ = self.lstm(mng_seq)
        last = out[:, -1, :]  # last hidden state [B, hidden_dim]
        state_p = self.state_proj(trade_state)
        x = torch.cat([last, state_p], dim=-1)
        logits = self.head(x)
        return logits


# --- Main script ---

# Load and preprocess data
df = fetch_mt5_data()
arr = preprocess(df)
# Scale data
scaler = MinMaxScaler()
arr_scaled = scaler.fit_transform(arr)

# Dataset configuration
input_len = 50
state_dim = 2  # Based on our dummy trade_state in the dataset
mng_feat_dim = arr_scaled.shape[1]

# Create a compatible dataset instance and DataLoader
dataset = TimeseriesDataset(arr_scaled, input_len=input_len)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model configuration, optimizer, and criterion
hidden_dim = 128
n_layers = 2
dropout = 0.2
model = LSTMManagement(mng_feat_dim, state_dim, hidden_dim, n_layers, dropout)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# The output is logits for Hold/Close, so use CrossEntropyLoss for classification
criterion = nn.CrossEntropyLoss()

# The training loop that correctly uses the dataset and model
print("Starting training...")
for epoch in range(3):
    for mng_seq, trade_state, labels in loader:
        # Get logits from the model
        logits = model(mng_seq, trade_state)
        # Calculate loss
        loss = criterion(logits, labels)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}, Loss {loss.item():.4f}")
print("Training complete.")

# Save artifacts
save_model(model, "checkpoints/management_lstm.pth")
save_scaler(scaler, "checkpoints/management_scaler.pkl")