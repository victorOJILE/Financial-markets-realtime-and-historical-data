import torch
import torch.nn as nn
import torch.optim as optim
from models.transformer_models import EntryModel
from utils.data_loader import get_dataloader

def train_entry_model():
    # Hyperparameter
    ltf_seq_length = 200 # Corresponds to 200 15M candles
    
    # Load data and HTF embeddings
    ltf_data, ltf_labels = torch.load('data/ltf_data.pt')
    htf_embeddings = torch.load('data/htf_embeddings.pt')
    
    # Create DataLoader for training
    # The dataloader handles batches of LTF sequences and their corresponding labels
    dataloader = get_dataloader(ltf_data, ltf_labels, ltf_seq_length)
    
    model = EntryModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training Loop
    model.train()
    for epoch in range(100):
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            # Transpose inputs for Transformer: (seq_length, batch_size, feature_dim)
            inputs = inputs.permute(1, 0, 2)
            
            # Align HTF embeddings with the current LTF batch
            # This is a crucial step. The index of the HTF embedding needs to be
            # calculated based on the position of the LTF sequence in the dataset.
            # Assuming a simple 1:N relationship (one 4H bar for every 16 15M bars)
            # and that data is aligned, we can find the start index of the HTF embedding
            # for the current LTF batch.
            htf_context_index = int((dataloader.batch_size * batch_idx + ltf_seq_length) / 16)
            
            # Select the correct HTF context for the entire batch
            # Ensure the index doesn't go out of bounds
            if htf_context_index < htf_embeddings.size(0):
                htf_context = htf_embeddings[htf_context_index:htf_context_index+1].expand(inputs.size(1), -1)
            else:
                # Handle edge case at end of dataset, e.g., use the last available embedding
                htf_context = htf_embeddings[-1:].expand(inputs.size(1), -1)
            
            optimizer.zero_grad()
            logits = model(inputs, htf_context)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        print(f"Entry Model - Epoch {epoch+1}, Loss: {loss.item():.4f}")

    # Save trained model
    torch.save(model.state_dict(), 'models/entry_model.pt')
    print("Entry Model training complete and saved.")

if __name__ == '__main__':
    train_entry_model()
