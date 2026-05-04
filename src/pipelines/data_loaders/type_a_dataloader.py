import torch
from torch.utils.data import Dataset
from pathlib import Path

class Dataset_A(Dataset):
    def __init__(self, images : torch.Tensor, sentences : torch.Tensor, transformation):
        # Dataset_A class adapted from:
        # Ugama Benedicta Kelechi (2025). 
        # Understanding PyTorch’s DataLoader: How to Efficiently Load and Augment Data. 
        # [online] Medium. Available at: https://medium.com/@ugamakelechi501/understanding-pytorchs-dataloader-how-to-efficiently-load-and-augment-data-c9eb26f61491.

        self.images = images
        self.sentences = sentences
        self.transformation = transformation
        if len(self.images) != len(self.sentences):
            raise ValueError(f'Mismatch in embedding lengths. IMG_EMB: {len(self.images)} | SENTENCE_EMB: {len(self.sentences)}')
        print('Dataloader initialisation complete')
        print(f'Sentence embeddings:\n>>> Shape: {self.sentences.shape}\n>>> Type: {self.sentences.dtype}')
        print(f'Image embeddings:\n>>> Shape: {self.images.shape}\n>>> Type: {self.images.dtype}')
                    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_emb = self.images[idx]
        if img_emb is None:
            raise ValueError(f'Img embedding for {idx} is none')
        img_emb_t = self.transformation(img_emb)
        sentence_emb = self.sentences[idx]
        return img_emb_t, sentence_emb