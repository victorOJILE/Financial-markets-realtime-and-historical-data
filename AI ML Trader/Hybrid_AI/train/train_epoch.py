from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

def train_one_epoch(model, dataloader, optimizer, criterion, scaler=None, max_grad_norm=1.0):
    model.train()
    epoch_loss = 0.0
    total = 0
    correct = 0
    for batch in tqdm(dataloader):
        # Example for Entry model: adapt keys as needed
        ltf = batch['ltf'].to(DEVICE)            # [B, T, F]
        htf = batch['htf_emb'].to(DEVICE)        # [B, emb_dim]
        labels = torch.tensor([x['entry'] for x in batch['labels']]).to(DEVICE)

        optimizer.zero_grad()
        with autocast(enabled=(scaler is not None)):
            logits = model(ltf, htf)
            loss = criterion(logits, labels)

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        epoch_loss += loss.item() * ltf.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += ltf.size(0)

    avg_loss = epoch_loss / total
    acc = correct / total
    return avg_loss, acc