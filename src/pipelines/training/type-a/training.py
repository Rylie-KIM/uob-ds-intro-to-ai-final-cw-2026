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
from src.pipelines.data_loaders.type_a_dataloader import Dataset_A
from src.models.CNN import CNN_encoder
from src.models.CNN2 import CNN2
from src.models.googleNet import GoogleNet
from pipelines.training.cosine_loss import CosineLoss

#####################################
######## Train Models ###############
#####################################

transform_fs = transforms.Compose([
                transforms.Resize((128,128)),
                transforms.Normalize((0.5, 0.5,0.5), (0.5, 0.5, 0.5))
        ])

transform_pt = transforms.Compose([
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

run_variations = [
        # Vary model size
    {'model' : CNN2Layer, 'loss': CosineLoss(), 'embedding': 'sbert_emb_master.pt', 'output_dims' : 384, 'transform' : transform_fs, 'emb' : 'sbert'},
    {'model' : CNNPool, 'loss': CosineLoss(), 'embedding': 'sbert_emb_master.pt', 'output_dims' : 384, 'transform' : transform_fs, 'emb' : 'sbert'},
    {'model' : CNNStride, 'loss': CosineLoss(), 'embedding': 'sbert_emb_master.pt', 'output_dims' : 384, 'transform' : transform_fs, 'emb' : 'sbert'},
    {'model' : GoogleNet, 'loss': CosineLoss(), 'embedding': 'sbert_emb_master.pt', 'output_dims' : 384, 'transform' : transform_pt, 'emb' : 'sbert'},
    {'model' : ResnetDSA, 'loss': CosineLoss(), 'embedding': 'sbert_emb_master.pt', 'output_dims' : 384, 'transform' : transform_pt, 'emb' : 'sbert'},
        
        # Vary Embeddings for cosine loss
    {'model' : ResnetDSA, 'loss': CosineLoss(), 'embedding': 'TB_pooler_emb_master.pt', 'output_dims' : 312, 'transform' : transform_pt, 'emb' : 'TB_pooler'},
    {'model' : ResnetDSA, 'loss': CosineLoss(), 'embedding': 'TB_mean_emb_master.pt', 'output_dims' : 312, 'transform' : transform_pt, 'emb' : 'TB_mean'},
    {'model' : ResnetDSA, 'loss': CosineLoss(), 'embedding': 'B_pooler_emb_master.pt', 'output_dims' : 768, 'transform' : transform_pt, 'emb' : 'B_pooler'},
    {'model' : ResnetDSA, 'loss': CosineLoss(), 'embedding': 'B_mean_emb_master.pt', 'output_dims' : 768, 'transform' : transform_pt, 'emb' : 'B_mean'},
    {'model' : ResnetDSA, 'loss': CosineLoss(), 'embedding': 'P_wvec_emb_master.pt', 'output_dims' : 300, 'transform' : transform_pt, 'emb' : 'P_wvec'},
    
        # Vary Embeddings with MSE loss
    {'model' : ResnetDSA, 'loss': nn.MSELoss(), 'embedding': 'sbert_emb_master.pt', 'output_dims' : 384, 'transform' : transform_pt, 'emb' : 'sbert'},
    {'model' : ResnetDSA, 'loss': nn.MSELoss(), 'embedding': 'TB_pooler_emb_master.pt', 'output_dims' : 312, 'transform' : transform_pt, 'emb' : 'TB_pooler'},
    {'model' : ResnetDSA, 'loss': nn.MSELoss(), 'embedding': 'TB_mean_emb_master.pt', 'output_dims' : 312, 'transform' : transform_pt, 'emb' : 'TB_mean'},
    {'model' : ResnetDSA, 'loss': nn.MSELoss(), 'embedding': 'B_pooler_emb_master.pt', 'output_dims' : 768, 'transform' : transform_pt, 'emb' : 'B_pooler'},
    {'model' : ResnetDSA, 'loss': nn.MSELoss(), 'embedding': 'B_mean_emb_master.pt', 'output_dims' : 768, 'transform' : transform_pt, 'emb' : 'B_mean'},
    {'model' : ResnetDSA, 'loss': nn.MSELoss(), 'embedding': 'P_wvec_emb_master.pt', 'output_dims' : 300, 'transform' : transform_pt, 'emb' : 'P_wvec'}
]
#   sentence_emb_stem = Path('C:/Masters/Text Analytics/AI_Coursework/embeddings')
#   img_emb_path = Path('C:/Masters/Text Analytics/AI_Coursework/embeddings/images_master.pt')

def train_val_test_split(images):
        torch.manual_seed(42)
        idxs = torch.randperm(len(images)).tolist()
        splits = {
                'train_idxs' : idxs[:8000],
                'val_idxs' : idxs[8000:9000],
                'test_idxs' : idxs[9000:]}
        return splits

def train_val_test_split(images):
    torch.manual_seed(42)
    idxs = torch.randperm(len(images)).tolist()
    splits = {
      'train_idxs' : idxs[:8000],
      'val_idxs' : idxs[8000:9000],
      'test_idxs' : idxs[9000:]}
    return splits

def run_cases(sentence_emb_stem : Path, images : torch.Tensor, run_variations : dict, save_to_path : Path):
        start = time.time()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not sentence_emb_stem.exists():
                raise FileNotFoundError(f'Sentence embeddings folder not found: {sentence_emb_stem}')
        print('Images file found')
        print('Sentence embeddings folder found.')
        splits = train_val_test_split(images)
        splits_file_path = save_to_path / 'splits.pt'
        torch.save(splits, splits_file_path)
        print(f'Splits saved successfully: {splits_file_path}')
        for idx, variation in enumerate(run_variations):
                sentence_emb_path = Path(sentence_emb_stem) / variation['embedding']
                if not sentence_emb_path.exists():
                        raise FileNotFoundError(f'Sentence embedding file not found: {sentence_emb_path}')
                sentences = torch.load(sentence_emb_path)
                dataset = Dataset_A(images, sentences, variation['transform'])
                training, validation = Subset(dataset, splits['train_idxs']), Subset(dataset, splits['val_idxs'])
                train_loader = DataLoader(training, batch_size = 32, shuffle=True)
                val_loader = DataLoader(validation, batch_size = 32, shuffle=False)
                model = variation['model']
                output_dims = variation['output_dims']
                model = model(output_dims = output_dims)
                model = model.to(device)
                print('-------------------------------------------------------')
                print(f'Beginning training for {model.__class__.__name__}')
                print(f'>> Embedding Type: {variation['emb']}')
                print(f'>> Loss: {variation['loss'].__class__.__name__}')
                train_losses, val_losses, best_model, total_time, loss_type = train(model, train_loader, val_loader, epochs = 15, learning_rate = 1e-5, criterion = variation['loss'])
                model_name = f"{idx}_{best_model.__class__.__name__}_{variation['emb']}_{variation['loss'].__class__.__name__}.pt"
                output_path = Path(save_to_path) / model_name
                best_model_data = {
                        'model_architecture': best_model.__class__.__name__,
                        'parameters': best_model.state_dict(),
                        'loss_type' : loss_type,
                        'embedding_type': variation['emb'],
                        'total_time': total_time, 
                        'train_losses' : train_losses,
                        'val_losses' : val_losses
                        }
                torch.save(best_model_data, output_path)
                print(f'Successfully trained {model_name} in {total_time:.2f}s')
                print(f'Saved model to: {output_path}')
        end = time.time() - start
        print(f'Model training has been finished in {end/60:.2f} minutes')

# run_cases(sentence_emb_stem, images, run_variations, output_path)