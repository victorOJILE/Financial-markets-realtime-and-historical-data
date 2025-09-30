import torch
import torch.nn as nn

class LSTMEntry(nn.Module):
 def __init__(self, input_dim=5, hidden_dim=64, num_layers=2, output_dim=3):
  super().__init__()
  self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
  self.fc = nn.Linear(hidden_dim, output_dim)

 def forward(self, x, context=None):
  out, _ = self.lstm(x)
  out = out[:, -1, :]
  if context is not None:
   out = torch.cat([out, context], dim=-1)
  return self.fc(out)






class LSTMEntry(nn.Module):
 def __init__(self, ltf_feat_dim, htf_emb_dim, hidden_dim=128, dropout=0.2, num_classes=3):
  super().__init__()
  self.lstm = nn.LSTM(input_size=ltf_feat_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True, dropout=dropout, bidirectional=False)
  self.h_proj = nn.Linear(htf_emb_dim, hidden_dim)
  self.head = nn.Sequential(
   nn.Linear(hidden_dim * 2, hidden_dim),
   nn.ReLU(),
   nn.LayerNorm(hidden_dim),
   nn.Dropout(dropout),
   nn.Linear(hidden_dim, num_classes)
  )

 def forward(self, ltf_seq, htf_emb):
  # ltf_seq: [B, T, F]; htf_emb: [B, htf_emb_dim]
  _, (h_n, _) = self.lstm(ltf_seq) # h_n: [num_layers, B, hidden_dim]
  last_h = h_n[-1] # [B, hidden_dim]
  htf_proj = self.h_proj(htf_emb)       # [B, hidden_dim]
  combined = torch.cat([last_h, htf_proj], dim=-1)  # [B, 2*hidden_dim]
  logits = self.head(combined)
  return logits
