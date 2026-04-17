import torch
import torch.optim as optim
from pipelines.training.earlyStopping import EarlyStopping
import time
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_one_epoch(model, train_set, optimizer, criterion):
    model.train()
    total_training_loss = 0.0
    for img_emb, sentence_emb in train_set:
        img_emb = img_emb.to(device)
        sentence_emb = sentence_emb.to(device)
        optimizer.zero_grad()
        outputs = model(img_emb)
        loss = criterion(outputs, sentence_emb)
        loss.backward()
        optimizer.step()
        total_training_loss += loss.item()
    return total_training_loss
    
def validation(model, val_loader, criterion):
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for img_emb, sentence_emb in val_loader:
            img_emb = img_emb.to(device)
            sentence_emb = sentence_emb.to(device)
            outputs = model(img_emb)
            loss = criterion(outputs, sentence_emb)
            total_val_loss += loss.item()
    return total_val_loss

def train(model, train_set, val_set, epochs, learning_rate, criterion):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience =2, min_lr = 1e-8
    )
    early_stopping = EarlyStopping(patience=4)
    train_losses = []
    val_losses = []
    start = time.time()
    for epoch in range(epochs):
        total_train_loss = train_one_epoch(model, train_set, optimizer, criterion)
        avg_train_loss = total_train_loss / len(train_set)
        train_losses.append(avg_train_loss)
        total_val_loss = validation(model, val_set, criterion)
        avg_val_loss = total_val_loss / len(val_set)
        scheduler.step(avg_val_loss)
        val_losses.append(avg_val_loss)
        early_stopping.check(avg_val_loss)
        print(f'Epoch: {epoch+1}/{epochs} | Train Loss: {round(avg_train_loss,6)} | Test Loss: {round(avg_val_loss,6)}')
        if early_stopping.stop == True:
            print('Early Stopping Activated!')
    total_time = time.time() - start
    loss_type = criterion.__class__.__name__
    return train_losses, val_losses, model, total_time, loss_type


    



