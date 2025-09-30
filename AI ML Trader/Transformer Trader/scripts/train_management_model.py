import torch
import torch.nn as nn
import torch.optim as optim
from models.transformer_models import ManagementModel
from utils.data_loader import get_dataloader

def train_management_model():
 # Hyperparameters
 seq_length = 100 # Corresponds to 100 15M candles
 
 # Load data for trade management
 # This data would be specific to segments where a trade was open.
 # We use LTF data as a placeholder here.
 ltf_data, management_labels = torch.load('data/ltf_data.pt')
 
 # Note: For real-world use, you would need to add trade-specific features
 # (e.g., PnL, elapsed time) to the input features array.
 
 dataloader = get_dataloader(ltf_data, management_labels, seq_length)
 model = ManagementModel()
 
 criterion = nn.CrossEntropyLoss()
 optimizer = optim.Adam(model.parameters(), lr=0.001)

 # Training Loop
 model.train()
 for epoch in range(100):
  for inputs, labels in dataloader:
   # Transpose inputs for Transformer
   inputs = inputs.permute(1, 0, 2)
   
   optimizer.zero_grad()
   logits = model(inputs)
   loss = criterion(logits, labels)
   loss.backward()
   optimizer.step()
  print(f"Management Model - Epoch {epoch+1}, Loss: {loss.item():.4f}")

 # Save trained model
 torch.save(model.state_dict(), 'models/management_model.pt')

if __name__ == '__main__':
 train_management_model()


