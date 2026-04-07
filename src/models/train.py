import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import torch.nn.functional as F
from CNN import CNN_encoder


def train(model, train_set, test_set, epochs, learning_rate):
    criterion = nn.CosineEmbeddingLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        model.train()
        train_running_loss = 0.0

        for img_emb, sentence_emb in train_set:

            optimizer.zero_grad()
            outputs = model(img_emb)
            outputs = F.normalize(outputs,p=2,dim=1)
            target = torch.ones(outputs.size(0))
            loss = criterion(outputs, sentence_emb, target)
            loss.backward()
            optimizer.step()
            train_running_loss += loss.item()

        avg_train_loss = train_running_loss / len(train_set)
        train_losses.append(avg_train_loss)

        model.eval()
        test_running_loss = 0.0

        with torch.no_grad():
            for img_emb, sentence_emb in test_set:
                outputs = model(img_emb)
                outputs = F.normalize(outputs,p=2, dim=1)
                target = torch.ones(outputs.size(0))
                test_loss = criterion(outputs, sentence_emb, target)
                test_running_loss += test_loss.item()

        avg_test_loss = test_running_loss / len(test_set)
        test_losses.append(avg_test_loss)

        print(f'Epoch: {epoch+1}/{epochs} | Train Loss: {round(avg_train_loss,2)} | Test Loss: {round(avg_test_loss,2)}')
   

    return train_losses, test_losses