import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np 
import math 
import pandas as pd
from pathlib import Path


class Dataset_A(Dataset):
    def __init__(self, embedding_type):
        # This needs to be updated to 'master.csv'
        self.df = pd.read_csv('./src/data/type-a/as_eps.csv')
        self.embedding_types = [
            'bert_pooler_embeddings',
            'bert_mean_embeddings',
            'sbert_embeddings',
            'word2vec_pretrained_embeddings']
        
        self.embedding_type = embedding_type
        self.root_path = Path('src/embeddings/computed-embeddings/type-a')
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = row['label']
        filename = row[self.embedding_type]
        embedding_path = self.root_path / self.embedding_type / filename
        embedding = torch.load(embedding_path)
        return embedding, label

# example_tensor = torch.randn(3,4) created 3 by 4 tensor of random values
# torch.save(example_tensor, 'example_tensor.pt')
# tensor_data = torch.load('example_tensor.pt')

# from torch.utils.data import DataLoader
# dataset = Dataset_A('bert_pooler_embeddings')
# dataset_loader = DataLoader(dataset, batch_size=32, shuffle=True)



