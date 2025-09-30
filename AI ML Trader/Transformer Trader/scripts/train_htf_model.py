import torch
import torch.nn as nn
import torch.optim as optim
from models.transformer_models import HTFContextModel
from utils.data_loader import get_dataloader

def train_htf_model():
 # Hyperparameters
 seq_length = 500 # Corresponds to 500 4H candles
 
 # Load data
 htf_data, htf_labels = torch.load('data/htf_data.pt')
 dataloader = get_dataloader(htf_data, htf_labels, seq_length)
 
 model = HTFContextModel()
 criterion = nn.CrossEntropyLoss()
 optimizer = optim.Adam(model.parameters(), lr=0.001)
 
 # Training Loop
 model.train()
 for epoch in range(100):
  for inputs, labels in dataloader:
   # Inputs shape: (batch_size, seq_length, feature_dim)
   # Need to transpose for Transformer: (seq_length, batch_size, feature_dim)
   inputs = inputs.permute(1, 0, 2)
   
   optimizer.zero_grad()
   logits, context_vector = model(inputs)
   loss = criterion(logits, labels)
   loss.backward()
   optimizer.step()
  print(f"HTF Model - Epoch {epoch}, Loss: {loss.item()}")

 # Save trained model and context embeddings
 torch.save(model.state_dict(), 'models/htf_context_model.pt')
 # Generate and save context embeddings for LTF training
 model.eval()
 with torch.no_grad():
  # Get all embeddings. This is a simplified approach. In reality, you'd use a dataloader
  # and process the entire dataset.
  _, htf_embeddings = model(htf_data.unsqueeze(1))
 torch.save(htf_embeddings, 'data/htf_embeddings.pt')

if __name__ == '__main__':
 train_htf_model()


def train_htf_model():
    # Create DataLoader for inference (to get embeddings)
    # Batch size can be larger during inference as we don't calculate gradients
    inference_dataloader = get_dataloader(htf_data, htf_labels, seq_length, batch_size=256, shuffle=False)
    
    # Set model to evaluation mode
    model.eval()
    
    all_embeddings = []
    with torch.no_grad():
        for inputs, _ in inference_dataloader:
            # Transpose inputs for Transformer
            inputs = inputs.permute(1, 0, 2)
            
            # Forward pass to get embeddings
            _, embeddings = model(inputs)
            
            # Append embeddings to a list
            all_embeddings.append(embeddings)
    
    # Concatenate all batches of embeddings into a single tensor
    htf_embeddings = torch.cat(all_embeddings, dim=0)
    
    # Save the final embeddings tensor
    torch.save(htf_embeddings, 'data/htf_embeddings.pt')
    print(f"Generated and saved {htf_embeddings.shape[0]} HTF embeddings.")

if __name__ == '__main__':
    train_htf_model()
