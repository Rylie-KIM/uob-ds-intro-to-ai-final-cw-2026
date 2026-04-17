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
            'total_time' : total_time,
            'train_losses' : train_losses,
            'val_losses' : val_losses
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

run_variations = [
        # Vary model size
    {'model' : CNN2Layer, 'loss': CosineLoss(), 'embedding': 'sbert_emb_master.pt', 'output_dims' : 384, 'transform' : transform_fs, 'emb' : 'sbert'},
    {'model' : CNNPool, 'loss': CosineLoss(), 'embedding': 'sbert_emb_master.pt', 'output_dims' : 384, 'transform' : transform_fs, 'emb' : 'sbert'},
    {'model' : CNNStride, 'loss': CosineLoss(), 'embedding': 'sbert_emb_master.pt', 'output_dims' : 384, 'transform' : transform_fs, 'emb' : 'sbert'},
    {'model' : GoogleNet, 'loss': CosineLoss(), 'embedding': 'sbert_emb_master.pt', 'output_dims' : 384, 'transform' : transform_pt, 'emb' : 'sbert'},
    {'model' : ResNet18Pretrained, 'loss': CosineLoss(), 'embedding': 'sbert_emb_master.pt', 'output_dims' : 384, 'transform' : transform_pt, 'emb' : 'sbert'},
        
        # Vary Embeddings for cosine loss
    {'model' : ResNet18Pretrained, 'loss': CosineLoss(), 'embedding': 'TB_pooler_emb_master.pt', 'output_dims' : 312, 'transform' : transform_pt, 'emb' : 'TB_pooler'},
    {'model' : ResNet18Pretrained, 'loss': CosineLoss(), 'embedding': 'TB_mean_emb_master.pt', 'output_dims' : 312, 'transform' : transform_pt, 'emb' : 'TB_mean'},
    {'model' : ResNet18Pretrained, 'loss': CosineLoss(), 'embedding': 'B_pooler_emb_master.pt', 'output_dims' : 768, 'transform' : transform_pt, 'emb' : 'B_pooler'},
    {'model' : ResNet18Pretrained, 'loss': CosineLoss(), 'embedding': 'B_mean_emb_master.pt', 'output_dims' : 768, 'transform' : transform_pt, 'emb' : 'B_mean'},
    {'model' : ResNet18Pretrained, 'loss': CosineLoss(), 'embedding': 'P_wvec_emb_master.pt', 'output_dims' : 300, 'transform' : transform_pt, 'emb' : 'P_wvec'},
    
        # Vary Embeddings with MSE loss
    {'model' : ResNet18Pretrained, 'loss': nn.MSELoss(), 'embedding': 'sbert_emb_master.pt', 'output_dims' : 384, 'transform' : transform_pt, 'emb' : 'sbert'},
    {'model' : ResNet18Pretrained, 'loss': nn.MSELoss(), 'embedding': 'TB_pooler_emb_master.pt', 'output_dims' : 312, 'transform' : transform_pt, 'emb' : 'TB_pooler'},
    {'model' : ResNet18Pretrained, 'loss': nn.MSELoss(), 'embedding': 'TB_mean_emb_master.pt', 'output_dims' : 312, 'transform' : transform_pt, 'emb' : 'TB_mean'},
    {'model' : ResNet18Pretrained, 'loss': nn.MSELoss(), 'embedding': 'B_pooler_emb_master.pt', 'output_dims' : 768, 'transform' : transform_pt, 'emb' : 'B_pooler'},
    {'model' : ResNet18Pretrained, 'loss': nn.MSELoss(), 'embedding': 'B_mean_emb_master.pt', 'output_dims' : 768, 'transform' : transform_pt, 'emb' : 'B_mean'},
    {'model' : ResNet18Pretrained, 'loss': nn.MSELoss(), 'embedding': 'P_wvec_emb_master.pt', 'output_dims' : 300, 'transform' : transform_pt, 'emb' : 'P_wvec'},

]
#   sentence_emb_stem = Path('C:/Masters/Text Analytics/AI_Coursework/embeddings')
#   img_emb_path = Path('C:/Masters/Text Analytics/AI_Coursework/embeddings/images_master.pt')

transform_fs = transforms.Compose([
                transforms.Resize((128,128)),
                transforms.Normalize((0.5, 0.5,0.5), (0.5, 0.5, 0.5))
        ])

transform_pt = transforms.Compose([
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

def run_cases(sentence_emb_stem:Path, img_emb_path:Path, run_variations:dict):
        start = time.time()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not sentence_emb_stem.exists():
              raise FileNotFoundError(f'Sentence embeddings folder not found: {sentence_emb_stem}')
        if not img_emb_path.exists():
              raise FileNotFoundError(f'Image embeddings file not found: {img_emb_path}')
        print('Images file found')
        print('Sentence embeddings folder found.')
        for variation in run_variations:
                sentence_emb_path = Path(sentence_emb_stem) / variation['embedding']
                if not sentence_emb_path.exists():
                      raise FileNotFoundError(f'Sentence embedding file not found: {sentence_emb_path}')
                print('Sentence embedding file found.')
                dataset = Dataset_A(img_emb_path, sentence_emb_path, transformation = variation['transform'])
                training, validation =  Subset(dataset, range(8000)), Subset(dataset, range(8000, 9000))
                train_loader = DataLoader(training, batch_size = 32, shuffle=False)
                test_loader = DataLoader(validation, batch_size = 32, shuffle=False)
                model = variation['model']
                output_dims = variation['output_dims']
                model = model(output_dims = output_dims)
                model = model.to(device)
                train_losses, val_losses, best_model, total_time, loss_type = train(model, train_loader, test_loader, epochs = 10, learning_rate = 0.00001, criterion = variation['loss'])
                model_name = f"{variation['model']}_{variation['emb']}.pt"
                output_path = Path(main_path) / model_name
                best_model_data = {
                        'model_architecture': variation['model'],
                        'parameters': best_model.state_dict(),
                        'loss_type' : loss_type,
                        'embedding_type': variation['emb'],
                        'total_time': total_time, 
                        'train_losses' : train_losses,
                        'val_losses' : val_losses
                        }
                torch.save(best_model_data, output_path)
                print(f'Successfully trained {model_name} in {total_time}')
                print(f'Saved model to: {output_path}')
        end = time.time() - start
        print(f'Model training has been finished in {end/60} minutes')
