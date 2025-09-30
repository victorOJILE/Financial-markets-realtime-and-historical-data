import torch
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
 """
 A custom PyTorch Dataset for time series data.
 """
 def __init__(self, features, labels, seq_length):
  self.features = features
  self.labels = labels
  self.seq_length = seq_length

 def __len__(self):
  return len(self.features) - self.seq_length + 1

 def __getitem__(self, idx):
  start_idx = idx
  end_idx = idx + self.seq_length
  return self.features[start_idx:end_idx], self.labels[end_idx-1]

def get_dataloader(features, labels, seq_length, batch_size, shuffle=True):
 """
 Creates a DataLoader for the given features and labels.
 """
 dataset = TimeSeriesDataset(features, labels, seq_length)
 return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
