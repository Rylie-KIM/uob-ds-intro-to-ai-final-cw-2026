import torch
from torch.utils.data import Dataset
from pathlib import Path


class Dataset_A(Dataset):
    def __init__(self, img_embs_path : Path, sentence_embs_path : Path, transformation ):

        self.sentence_embs_path = Path(sentence_embs_path)
        if not self.sentence_embs_path.exists():
            raise FileNotFoundError(f'PATH NOT FOUND: {self.sentence_embs_path}')
        self.img_embs_path = Path(img_embs_path)
        if not self.img_embs_path.exists():
            raise FileNotFoundError(f'PATH NOT FOUND: {self.img_embs_path}')
        self.sentence_embs = torch.load(self.sentence_embs_path)
        self.img_embs= torch.load(self.img_embs_path)
        self.transformation = transformation
        if len(self.img_embs) != len(self.sentence_embs):
            raise ValueError(f'Mismatch in embedding lengths. IMG_EMB: {len(self.img_embs)} | SENTENCE_EMB: {len(self.sentence_embs)}')
        print('Dataloader initialisation complete')
        print(f'Sentence embeddings:\n>>> Shape: {self.sentence_embs.shape}\n>>> Type: {self.sentence_embs.dtype}')
        print(f'Image embeddings:\n>>> Shape: {self.img_embs.shape}\n>>> Type: {self.img_embs.dtype}')
                    
    def __len__(self):
        return len(self.img_embs)

    def __getitem__(self, idx):
        img_emb = self.img_embs[idx]
        if img_emb is None:
            raise ValueError(f'Img embedding for {idx} is none')
        img_emb_t = self.transformation(img_emb)
        sentence_emb = self.sentence_embs[idx]
        return img_emb_t, sentence_emb