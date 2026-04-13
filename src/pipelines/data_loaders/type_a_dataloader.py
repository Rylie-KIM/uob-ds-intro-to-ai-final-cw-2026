import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import torch.nn.functional as F
import torchvision.transforms as transforms


class Dataset_A(Dataset):
    def __init__(self, embedding_type:str, img_embs_path, sentence_embs_path ):
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
        self.sentence_embs_path = Path(sentence_embs_path)
        if not self.sentence_embs_path.exists():
            raise FileNotFoundError(f'PATH NOT FOUND: {self.sentence_embs_path}')
        self.img_embs_path = Path(img_embs_path)
        if not self.img_embs_path.exists():
            raise FileNotFoundError(f'PATH NOT FOUND: {self.img_embs_path}')
        self.sentence_embs = torch.load(self.sentence_embs_path)
        self.img_embs= torch.load(self.img_embs_path)
        self.sentence_emb_type = embedding_type
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
        sentence_emb = self.sentence_embs[idx]
        sentence_embedding_norm = F.normalize(sentence_emb.float(), p=2, dim=0)
        return img_emb, sentence_embedding_norm


# example_tensor = torch.randn(3,4) created 3 by 4 tensor of random values
# torch.save(example_tensor, 'example_tensor.pt')
# tensor_data = torch.load('example_tensor.pt')