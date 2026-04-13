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


"""
For googlenet model:
weights = GoogLeNet_Weights.DEFAULT
transform = weights.transforms()

I have pre transformed the data in one_emb so if i need different sizes will have to edit that

"""

embedding_types = ['TB_pooler_emb','TB_mean_emb','B_pooler_emb','B_mean_emb','sbert_emb']
embedding_type = 'TB_pooler_emb'
img_emb_path = Path("C:/Masters/Text Analytics/AI_Coursework/embeddings/images_master.pt")
sentence_emb_path = Path("C:/Masters/Text Analytics/AI_Coursework/embeddings/TB_pooler_emb_master.pt")
dataset = Dataset_A('TB_pooler', img_emb_path, sentence_emb_path)
training, testing = Subset(dataset, range(16000)), Subset(dataset, range(16000, len(dataset)))
bs = 32
train_loader = DataLoader(training, batch_size = bs, shuffle=True)
# Dont shuffle test set so you get consistent eval
test_loader = DataLoader(testing, batch_size = bs, shuffle=False)
# print(f'Train_loader length:\n  >>Batches: {len(train_loader)}\n  >>Size: {len(train_loader) * 32}')
# print(f'Test_loader length:\n  >>Batches: {len(test_loader)}\n  >>Size: {len(test_loader) * 32}')

model = CNN_encoder(embedding_dims = 312)
train_losses, test_losses, best_model = train(model, train_set = train_loader, test_set = test_loader, epochs = 10, learning_rate = 0.000001)
main_path = Path("C:/Masters/Text Analytics/AI_Coursework/trained_models")
model_name = f'{best_model.__class__.__name__}.pt'
file_path = main_path / model_name
torch.save(best_model.state_dict(), file_path)
#plt.plot(train_losses)
#plt.plot(test_losses)
#plt.show()