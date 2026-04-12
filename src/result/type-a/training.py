import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(_ROOT))
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data import Subset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.models.train import train
from src.pipelines.data_loaders.type_a_dataloader import Dataset_A
from src.models.CNN import CNN_encoder

transform = transforms.Compose([
    # May want to vary the resizing and see how that effects the accuracy
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

"""
For googlenet model:
weights = GoogLeNet_Weights.DEFAULT
transform = weights.transforms()

"""
embedding_types = ['TB_pooler_emb','TB_mean_emb','B_pooler_emb','B_mean_emb','sbert_emb']
embedding_type = 'TB_pooler_emb'
dataset = Dataset_A(embedding_type, transform_imgs = transform)
training, testing = Subset(dataset, range(16000)), Subset(dataset, range(16000, len(dataset)))
train_loader = DataLoader(training, batch_size = 1, shuffle=True)
# Dont shuffle test set so you get consistent eval
test_loader = DataLoader(testing, batch_size = 1, shuffle=False)
# print(f'Train_loader length:\n  >>Batches: {len(train_loader)}\n  >>Size: {len(train_loader) * 32}')
# print(f'Test_loader length:\n  >>Batches: {len(test_loader)}\n  >>Size: {len(test_loader) * 32}')

model = CNN_encoder(embedding_dims = 312)
print('here')
train_losses, test_losses = train(model, train_set = train_loader, test_set = test_loader, epochs = 30, learning_rate = 0.003, weight_decay = 0.001)

plt.plot(train_losses)
plt.plot(test_losses)