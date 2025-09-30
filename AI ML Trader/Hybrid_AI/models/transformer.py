import torch
import torch.nn as nn

class TransformerModel(nn.Module):
 def __init__(self, feature_size=5, num_heads=4, num_layers=2, hidden_dim=128):
  super().__init__()
  self.embedding = nn.Linear(feature_size, hidden_dim)
  encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
  self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
  self.fc = nn.Linear(hidden_dim, hidden_dim)

 def forward(self, x):
  x = self.embedding(x)
  x = self.transformer(x)
  return self.fc(x.mean(dim=1))
  
  
  
class SimplePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1), :]

class HTFTransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model=128, n_heads=4, n_layers=3, ff_dim=256, dropout=0.1, out_dim=64):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos = SimplePositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, ff_dim, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out = nn.Sequential(nn.Linear(d_model, out_dim), nn.LayerNorm(out_dim), nn.ReLU())

    def forward(self, x):  # x: [B, T, F]
        x = self.input_proj(x)          # [B, T, d_model]
        x = self.pos(x)
        x = self.encoder(x)             # [B, T, d_model]
        # pool over time
        x = x.transpose(1, 2)           # [B, d_model, T]
        x = self.pool(x).squeeze(-1)    # [B, d_model]
        emb = self.out(x)               # [B, out_dim]
        return emb