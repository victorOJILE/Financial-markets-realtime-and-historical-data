import torch
from torch.utils.data import DataLoader
from data.mt5_data import fetch_mt5_data, preprocess
from data.dataset import TimeseriesDataset
from models.transformer import TransformerModel
from utils.save_utils import save_model, save_scaler
from sklearn.preprocessing import MinMaxScaler

# Load data
df = fetch_mt5_data("H4", n_bars=5000)
arr = preprocess(df)
# Scale
scaler = MinMaxScaler()
arr_scaled = scaler.fit_transform(arr)
# Dataset
dataset = TimeseriesDataset(arr_scaled, input_len=200, forecast_horizon=5)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
# Model
model = TransformerModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()

for epoch in range(10):
    for X, y in loader:
        out = model(X)
        loss = criterion(out.squeeze(), y.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}, Loss {loss.item():.4f}")

# Save artifacts
save_model(model, "checkpoints/transformer_model.pth")
save_scaler(scaler, "checkpoints/transformer_scaler.pkl")
