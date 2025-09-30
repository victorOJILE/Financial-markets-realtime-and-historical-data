import torch
import torch.nn as nn

class LSTMManagement(nn.Module):
 def __init__(self, input_dim=5, hidden_dim=64, num_layers=2, output_dim=3):
  super().__init__()
  self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
  self.fc = nn.Linear(hidden_dim, output_dim)

 def forward(self, x, pos_info=None):
  out, _ = self.lstm(x)
  out = out[:, -1, :]
  if pos_info is not None:
   out = torch.cat([out, pos_info], dim=-1)
  return self.fc(out)




class LSTMManagement(nn.Module):
 def __init__(self, mng_feat_dim, state_dim, hidden_dim=128, n_layers=2, dropout=0.2):
  super().__init__()
  self.lstm = nn.LSTM(mng_feat_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout)
  self.state_proj = nn.Linear(state_dim, hidden_dim)
  self.head = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2)) # Hold/Close

 def forward(self, mng_seq, trade_state):
  # mng_seq: [B, T, F]; we may want logits for each timestep -> return last-step decision or all steps
  out, _ = self.lstm(mng_seq)
  last = out[:, -1, :] # last hidden state [B, hidden_dim]
  state_p = self.state_proj(trade_state)
  x = torch.cat([last, state_p], dim=-1)
  logits = self.head(x)
  return logits
