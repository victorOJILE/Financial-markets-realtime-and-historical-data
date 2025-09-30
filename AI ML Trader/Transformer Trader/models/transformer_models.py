import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from scripts.data_preparation import features

class HTFContextModel(nn.Module):
 """
 Transformer model for generating HTF context embeddings.
 """
 def __init__(self):
  super().__init__()
  feature_dim = len(features)
  d_model = 128
  nhead = 4
  num_layers = 2
  num_classes = 3 # Up/Down/Sideways
  
  self.encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4)
  self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers)
  self.embedding = nn.Linear(feature_dim, d_model)
  self.pos_encoder = PositionalEncoding(d_model)
  self.fc = nn.Linear(d_model, num_classes) # For pre-training

 def forward(self, src):
  src = self.embedding(src)
  src = self.pos_encoder(src)
  output = self.transformer_encoder(src)
  # We take the embedding of the last token as the context vector
  context_vector = output[-1, :, :]
  logits = self.fc(context_vector)
  return logits, context_vector # Return both logits and embedding

class EntryModel(nn.Module):
 """
 Transformer model for generating entry signals, takes HTF context as input.
 """
 def __init__(self):
  super().__init__()
  feature_dim = len(features)
  htf_context_dim = 128
  d_model = 128
  nhead = 4
  num_layers = 2
  num_classes = 3

  self.encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4)
  self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers)
  self.ltf_embedding = nn.Linear(feature_dim, d_model)
  self.htf_context_proj = nn.Linear(htf_context_dim, d_model)
  self.pos_encoder = PositionalEncoding(d_model)
  self.fc = nn.Linear(d_model, num_classes)

 def forward(self, ltf_src, htf_context_vec):
  ltf_emb = self.ltf_embedding(ltf_src)
  htf_emb = self.htf_context_proj(htf_context_vec).unsqueeze(0)
  # Concatenate HTF context with LTF sequence
  combined_src = torch.cat([htf_emb, ltf_emb], dim=0)
  combined_src = self.pos_encoder(combined_src)
  output = self.transformer_encoder(combined_src)
  # Decision based on the first token (context) or last token of the sequence
  decision_vector = output[0, :, :]
  logits = self.fc(decision_vector)
  return logits

class ManagementModel(nn.Module):
 """
 Transformer model for managing open trades.
 """
 def __init__(self):
  super().__init__()
  feature_dim = len(features)
  d_model = 128
  nhead = 4
  num_layers = 2
  num_classes = 2 # Close/Hold
  
  self.encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4)
  self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers)
  self.embedding = nn.Linear(feature_dim, d_model)
  self.pos_encoder = PositionalEncoding(d_model)
  self.fc = nn.Linear(d_model, num_classes)

 def forward(self, src):
  src = self.embedding(src)
  src = self.pos_encoder(src)
  output = self.transformer_encoder(src)
  decision_vector = output[-1, :, :]
  logits = self.fc(decision_vector)
  return logits

class PositionalEncoding(nn.Module):
 def __init__(self, d_model, dropout=0.1, max_len=5000):
  super().__init__()
  self.dropout = nn.Dropout(p=dropout)
  pe = torch.zeros(max_len, d_model)
  position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
  div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
  pe[:, 0::2] = torch.sin(position * div_term)
  pe[:, 1::2] = torch.cos(position * div_term)
  pe = pe.unsqueeze(0).transpose(0, 1)
  self.register_buffer('pe', pe)

 def forward(self, x):
  x = x + self.pe[:x.size(0), :]
  return self.dropout(x)
