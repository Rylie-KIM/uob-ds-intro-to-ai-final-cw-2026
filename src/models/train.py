import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import torch.nn.functional as F
from src.models.earlyStopping import EarlyStopping
import matplotlib.pyplot as plt
import time
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
    # criterion = nn.CosineEmbeddingLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay = weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    ) 

"""


def train(model, train_set, test_set, epochs, learning_rate):
    # criterion = nn.CosineEmbeddingLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    early_stopping = EarlyStopping(patience=4)

    train_losses = []
    test_losses = []
    start = time.time()
    for epoch in range(epochs):
        print('starting new epoch...')
        model.train()
        train_running_loss = 0.0

        for img_emb, sentence_emb in train_set:

            img_emb = img_emb.to(device)
            sentence_emb = sentence_emb.to(device)

            optimizer.zero_grad()
            outputs = model(img_emb)
            #outputs = F.normalize(outputs,p=2,dim=1)
            #target = torch.ones(outputs.size(0)).to(device)
            loss = (1- F.cosine_similarity(outputs, sentence_emb, dim=1)).mean()
            # loss = criterion(outputs, sentence_emb, target)
            loss.backward()
            optimizer.step()
            train_running_loss += loss.item()

        avg_train_loss = train_running_loss / len(train_set)
        train_losses.append(avg_train_loss)

        model.eval()
        test_running_loss = 0.0

        with torch.no_grad():
            for img_emb, sentence_emb in test_set:
                img_emb = img_emb.to(device)
                sentence_emb = sentence_emb.to(device)

                outputs = model(img_emb)
                # normalising output - may want to check that
                #outputs = F.normalize(outputs,p=2, dim=1)
                #target = torch.ones(outputs.size(0)).to(device)
                test_loss = (1- F.cosine_similarity(outputs, sentence_emb, dim=1)).mean()
                # test_loss = criterion(outputs, sentence_emb, target)
                test_running_loss += test_loss.item()
        print('computing averages...')
        avg_test_loss = test_running_loss / len(test_set)
        test_losses.append(avg_test_loss)
        early_stopping.check(avg_test_loss)
        if early_stopping.stop == True:
            print('Early Stopping Activated!')
            print(f'Epoch: {epoch+1}/{epochs} | Train Loss: {round(avg_train_loss,4)} | Test Loss: {round(avg_test_loss,4)}')
            break
        print(f'Epoch: {epoch+1}/{epochs} | Train Loss: {round(avg_train_loss,4)} | Test Loss: {round(avg_test_loss,4)}')
    total_time = time.time() - start
    return train_losses, test_losses, model, total_time