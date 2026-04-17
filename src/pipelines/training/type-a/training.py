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
import time
import matplotlib.pyplot as plt
from src.models.train import train
from src.pipelines.data_loaders.type_a_dataloader import Dataset_A
from src.models.CNN import CNN_encoder
from src.models.CNN2 import CNN2
from src.models.googleNet import GoogleNet
from pipelines.training.cosine_loss import CosineLoss


"""
For googlenet model:
weights = GoogLeNet_Weights.DEFAULT
transform = weights.transforms()

I have pre transformed the data in one_emb so if i need different sizes will have to edit that

Embeddings:
>>TB_pooler = 312
>>TB_mean = 312
>>B_pooler = 768
>>B_mean = 768
>>sbert = 384
>>P_wvec = 300

Models:
>>CNN_encoder(output_dims = ANY)
>>CNN2(output_dims_dims = ANY)
>>GoogleNet(output_dims_dims = ANY) BUT shape == [batch_size, 3, 224, 224]
>>alexnet(output_dims_dims = ANY) BUT shape == [batch_size, 3, 224, 224]
"""
"""
For cnn_encoder and CNN2, resize to 128,128 and normalize normally
for googlenet, alexnet, apply individual normalisation as size is correct
"""
embedding_types = ['TB_pooler_emb','TB_mean_emb','B_pooler_emb','B_mean_emb','sbert_emb', 'P_wvec']

transform_fs = transforms.Compose([
                transforms.Resize((128,128)),
                transforms.Normalize((0.5, 0.5,0.5), (0.5, 0.5, 0.5))
])

transform_pt = transforms.Compose([
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

sentence_embs_dict = {
            'TB_pooler_emb':(312, Path("C:/Masters/Text Analytics/AI_Coursework/embeddings/TB_pooler_emb_master.pt")),
            'TB_mean_emb': (312, Path("C:/Masters/Text Analytics/AI_Coursework/embeddings/TB_mean_emb_master.pt")),
            'B_pooler_emb': (768, Path("C:/Masters/Text Analytics/AI_Coursework/embeddings/B_pooler_emb_master.pt")),
            'B_mean_emb': (768, Path("C:/Masters/Text Analytics/AI_Coursework/embeddings/B_mean_emb_master.pt")),
            'sbert_emb': (384, Path("C:/Masters/Text Analytics/AI_Coursework/embeddings/sbert_emb_master.pt")),
            'P_wvec': (300, Path("C:/Masters/Text Analytics/AI_Coursework/embeddings/P_wvec_emb_master.pt"))
        }

img_emb_path = Path("C:/Masters/Text Analytics/AI_Coursework/embeddings/images_master.pt")

################################ 
################################ 
###### From Scrtach Models###### 
################################ 
################################ 

models = [CNN_encoder, CNN2]
bs = 32
main_path = Path("C:/Masters/Text Analytics/AI_Coursework/trained_models")
for model_type in models:
    for emb in embedding_types:
        sentence_emb_path = sentence_embs_dict[emb][1]
        dataset = Dataset_A(emb, img_emb_path, sentence_emb_path, transformation = transform_fs)
        training, validation =  Subset(dataset, range(8000)), Subset(dataset, range(8000, 9000))
        train_loader = DataLoader(training, batch_size = bs, shuffle=False)
        test_loader = DataLoader(validation, batch_size = bs, shuffle=False)
        emb_dims = sentence_embs_dict[emb][0]
        print(f'Training {model_type.__name__} with {emb}.....')
        model = model_type(output_dims=emb_dims)
        train_losses, val_losses, best_model, total_time, loss_type = train(model, train_set = train_loader, test_set = test_loader, epochs = 10, learning_rate = 0.000001, criterion = CosineLoss())
        model_name = f'{best_model.__class__.__name__}_{emb}.pt'
        output_path = main_path / model_name

        best_model_data = {
            'model_architecture': best_model.__class__.__name__,
            'parameters': best_model.state_dict(),
            'loss_type' : loss_type,
            'embedding_type': emb,
            'total_time' : total_time
        }
        torch.save(best_model_data, output_path)
        print(f'Successfully trained {model_name} in {total_time}')
        print(f'Saved model to: {output_path}')



################################ 
################################ 
###### Pre-trained Models ###### # training googlenet took approx 6-7 mins per epoch
################################ 
################################ 

sentence_emb_path = Path("C:/Masters/Text Analytics/AI_Coursework/embeddings/sbert_emb_master.pt")
bs = 32
model = GoogleNet(output_dims = 384)
dataset = Dataset_A('sbert_emb', img_emb_path, sentence_emb_path, transformation = transform_pt)
training, validation =  Subset(dataset, range(8000)), Subset(dataset, range(8000, 9000))
train_loader = DataLoader(training, batch_size = bs, shuffle=False)
test_loader = DataLoader(validation, batch_size = bs, shuffle=False)
train_losses, val_losses, best_model, time = train(model, train_set = train_loader, test_set = test_loader, epochs = 10, learning_rate = 0.000001)


#####################################
######## Train GoogleNet#############
#####################################

