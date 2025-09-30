import torch
from torch.utils.data import Dataset

# Delete this
class TimeseriesDataset(Dataset):
    def __init__(self, arr, input_len=100, forecast_horizon=1):
        self.arr = arr
        self.input_len = input_len
        self.forecast_horizon = forecast_horizon
        self.X, self.y = self._create_windows()

    def _create_windows(self):
        X, y = [], []
        for i in range(len(self.arr) - self.input_len - self.forecast_horizon):
            X.append(self.arr[i:i+self.input_len])
            y.append(self.arr[i+self.input_len+self.forecast_horizon-1][3])  # close price target
        return torch.tensor(X), torch.tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class TimeseriesDataset(Dataset):
    def __init__(self, data, input_len=50):
        self.data = data
        self.input_len = input_len
        self.samples = self._create_samples()

    def _create_samples(self):
        samples = []
        for i in range(len(self.data) - self.input_len - 1):
            # Input sequence (mng_seq)
            mng_seq = self.data[i:i + self.input_len]
            # Trade state (e.g., last price, profit/loss, etc.)
            # For this example, let's use the last two features of the sequence as the trade state
            trade_state = self.data[i + self.input_len - 1, -2:]
            # Label: a simplified binary label
            # 1 if the next close price is higher, 0 otherwise
            label = 1 if self.data[i + self.input_len, 0] > self.data[i + self.input_len - 1, 0] else 0
            samples.append({'mng_seq': mng_seq, 'trade_state': trade_state, 'label': label})
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        mng_seq = torch.tensor(sample['mng_seq'], dtype=torch.float32)
        trade_state = torch.tensor(sample['trade_state'], dtype=torch.float32)
        label = torch.tensor(sample['label'], dtype=torch.long)
        return mng_seq, trade_state, label
