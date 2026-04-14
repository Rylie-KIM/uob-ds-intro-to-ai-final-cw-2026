import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np 
import pandas as pd
from pathlib import Path
from PIL import Image
import sys
import torch.nn.functional as F
import torchvision.transforms as transforms


class Dataset_A(Dataset):
    def __init__(self, embedding_type:str, transform_imgs:type):
        """ 
            transform = transforms.Compose([
                transforms.Resize((128,128)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        """
        """
        Returns image embedding - Normalised, as tensor, and resized
        Returns sentence embedding, as tensor, normalized
        
        """
        self._ROOT = Path(__file__).resolve().parent.parent.parent.parent
        # This needs to be updated to 'master.csv'
        self.csv_path = self._ROOT / 'src' / 'data' / 'type-a' / 'master.csv'
        if not self.csv_path.exists():
            raise FileNotFoundError(f'PATH NOT FOUND: {self.csv_path}')
        
        self.df = pd.read_csv(self.csv_path)
        self.sentence_embedding_types = [
            #'OneHot_emb',
            'TB_pooler_emb',
            'TB_mean_emb',
            'B_pooler_emb',
            'B_mean_emb',
            'sbert_emb'
            ]
        
        self.sentence_embedding_type = embedding_type
        if self.sentence_embedding_type not in self.sentence_embedding_types:
            raise ValueError(f'{self.sentence_embedding_type} not in {self.sentence_embedding_types}')
        self.transform_imgs = transform_imgs
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        
        row = self.df.iloc[idx]
        

        # Sentence EMB Retrival
        sentence_emb_filename = row[self.sentence_embedding_type].replace('\\', '/')
        sentence_embedding_path = self._ROOT / Path(sentence_emb_filename)
        if not sentence_embedding_path.exists():
            raise FileNotFoundError(f'PATH NOT FOUND: {sentence_embedding_path}')
        sentence_embedding = torch.load(sentence_embedding_path)
        if not isinstance(sentence_embedding, torch.Tensor):
            raise TypeError(f'Sentence Embedding is not type tensor:\n{type(sentence_embedding)}')
        # Normalize sentence embeddings
        sentence_embedding_norm = F.normalize(sentence_embedding, p=2, dim=0)

        #Image EMB Retrival
        img_filename = row['path'].replace('\\', '/')
        
        img_path = self._ROOT / Path(img_filename)
        if not img_path.exists():
            raise FileNotFoundError(f'PATH NOT FOUND: {img_path}')
        
        img = Image.open(img_path).convert('RGB')
        img_emb = self.transform_imgs(img)
        return img_emb, sentence_embedding_norm

# example_tensor = torch.randn(3,4) created 3 by 4 tensor of random values
# torch.save(example_tensor, 'example_tensor.pt')
# tensor_data = torch.load('example_tensor.pt')

from torch.utils.data import DataLoader

transform = transforms.Compose([
                transforms.Resize((128,128)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
dataset = Dataset_A('TB_pooler_emb', transform_imgs =transform)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

import time 
start = time.time()
x = 0
for img, sent in dataloader:
    print(img.shape, sent.shape)
    print('done')
    x+=1
    if x>20:
        break
total = time.time() - start
print(total)
# dataset_loader = DataLoader(dataset, batch_size=32, shuffle=True)

