import torch
import torch.nn as nn

MAX_CANDLES = 50
FEATURES = 5
LABELS = ["Buy", "Sell", "Hold"]

class PositionalEncoding(nn.Module):
 def __init__(self, d_model, max_len=MAX_CANDLES):
  super().__init__()
  pe = torch.zeros(max_len, d_model)
  position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
  div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
  pe[:, 0::2] = torch.sin(position * div_term)
  pe[:, 1::2] = torch.cos(position * div_term)
  self.pe = pe.unsqueeze(0)

 def forward(self, x):
  x = x + self.pe[:, :x.size(1), :].to(x.device)
  return x

class SMTTransformer(nn.Module):
 def __init__(self, d_model=64, nhead=4, num_layers=2):
  super().__init__()
  self.input_proj = nn.Linear(FEATURES, d_model)
  self.position_encoding = PositionalEncoding(d_model)
  encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
  self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
  self.cross_attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
  self.classifier = nn.Sequential(
   nn.Linear(d_model, 64),
   nn.ReLU(),
   nn.Linear(64, len(LABELS))
  )

 def forward(self, x):
  b, p, t, c, f = x.shape
  x = x.view(b * p * t, c, f)
  x = self.input_proj(x)
  x = self.position_encoding(x)
  x = self.encoder(x)
  x = x.mean(dim=1)
  x = x.view(b, p * t, -1)
  pair_1 = x[:, :t, :]
  pair_2 = x[:, t:, :]
  attn_output, _ = self.cross_attention(pair_1, pair_2, pair_2)
  final_rep = (pair_1 + attn_output).mean(dim=1)
  return self.classifier(final_rep)

