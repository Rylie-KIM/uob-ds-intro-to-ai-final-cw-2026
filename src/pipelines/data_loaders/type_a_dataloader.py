import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np 
import pandas as pd
from pathlib import Path
from PIL import Image
import sys
import torch.nn.functional as F
import torchvision.transforms as transforms
import time


class Dataset_A(Dataset):
    def __init__(self, embedding_type:str):
        """ 
            transform = transforms.Compose([
                transforms.Resize((128,128)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]) - Transformation already done in one_emb file
        """
        """
        Returns image embedding - Normalised, as tensor, and resized
        Returns sentence embedding, as tensor, normalized
        
        """
        self._ROOT = Path(__file__).resolve().parent.parent.parent.parent
        self.sentence_embedding_types = [
            'TB_pooler_emb',
            'TB_mean_emb',
            'B_pooler_emb',
            'B_mean_emb',
            'sbert_emb'
            ]
        self.sentence_emb_paths = {
            'TB_pooler_emb': self._ROOT / 'src' / 'embeddings' / 'computed-embeddings' / 'type-a' / 'results' / 'TB_pooler_emb_master.pt',
            'TB_mean_emb': self._ROOT / 'src' / 'embeddings' / 'computed-embeddings' / 'type-a' / 'results' / 'TB_mean_emb_master.pt',
            'B_pooler_emb': self._ROOT / 'src' / 'embeddings' / 'computed-embeddings' / 'type-a' / 'results' / 'B_pooler_emb_master.pt',
            'B_mean_emb': self._ROOT / 'src' / 'embeddings' / 'computed-embeddings' / 'type-a' / 'results' / 'B_mean_emb_master.pt',
            'sbert_emb': self._ROOT / 'src' / 'embeddings' / 'computed-embeddings' / 'type-a' / 'results' / 'sbert_emb_master.pt'
        }
                    
        self.sentence_embedding_type = embedding_type
        if self.sentence_embedding_type not in self.sentence_embedding_types:
            raise ValueError(f'{self.sentence_embedding_type} not in {self.sentence_embedding_types}')
        
        self.sentence_embeddings_path = self.sentence_emb_paths[self.sentence_embedding_type]
        if not self.sentence_embeddings_path.exists():
            raise FileNotFoundError(f'PATH NOT FOUND: {self.sentence_embeddings_path}')
        self.sentence_embs = torch.load(self.sentence_embeddings_path, weights_only=True)

        self.image_embeddings_path = self._ROOT / 'src' / 'embeddings' / 'computed-embeddings' / 'type-a' / 'results' / 'images_master.pt'
        if not self.image_embeddings_path.exists():
            raise FileNotFoundError(f'PATH NOT FOUND: {self.image_embeddings_path}')
        
        self.image_embs= torch.load(self.image_embeddings_path, weights_only=True)
        print(f'Sentence embeddings:\n>>> Shape: {self.sentence_embs.shape}\n>>> Type: {self.sentence_embs.dtype}')
        print(f'Image embeddings:\n>>> Shape: {self.image_embs.shape}\n>>> Type: {self.image_embs.dtype}')
    
    def __len__(self):
        return len(self.image_embs)

    def __getitem__(self, idx):
        img_emb = self.image_embs[idx]
        if img_emb is None:
            raise ValueError(f'Img embedding for {idx} is none')
        sentence_emb = self.sentence_embs[idx]
        sentence_embedding_norm = F.normalize(sentence_emb.float(), p=2, dim=0)
        return img_emb, sentence_embedding_norm


# example_tensor = torch.randn(3,4) created 3 by 4 tensor of random values
# torch.save(example_tensor, 'example_tensor.pt')
# tensor_data = torch.load('example_tensor.pt')