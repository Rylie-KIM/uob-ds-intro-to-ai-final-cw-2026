import numpy as np
import numpy.typing as npt
from typing import TypeAlias
from sentence_transformers import SentenceTransformer
from pathlib import Path
import pandas as pd
import torch

# Type Aliases
Sentences: TypeAlias = list[str]
EmbeddingMatrix: TypeAlias = npt.NDArray[np.float32]

class SBERTEmbedder:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu"
    ):
        self.model_name = model_name
        self.device = device
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.master_path = Path('src/data/type-a/master.csv')
        self.folder_path = Path('src/embeddings/computed-embeddings/type-a/sbert')
        self.folder_path.mkdir(parents=True, exist_ok=True)
        self.df = pd.read_csv(self.master_path)

    def transform(self, sentence: str, idx:int):
        
        if self.model is None:
            raise ValueError("Call fit() before transform()")

        embeddings = self.model.encode(
            # changed to sentence
            sentence,
            convert_to_numpy=True,
            # Unsure if we need to normalize this embedding
            normalize_embeddings=True
        )
        embeddings = embeddings.astype(np.float32)
        filename = f'{idx}_sbert.pt'
        filepath = self.folder_path / filename
        torch.save(torch.tensor(embeddings), filepath)
        print(f'File: {filename}\nSaved to: {self.folder_path}')
        return filepath

    def process(self):
        filepaths = []
        for idx, row in self.df.iterrows():
            sentence = row['label']
            filepaths.append(self.transform(sentence, idx))
        self.df['sbert_emb'] = filepaths
        print(f'Success.{len(filepaths)} filepaths added')
        self.df.to_csv(self.master_path, index=False)

