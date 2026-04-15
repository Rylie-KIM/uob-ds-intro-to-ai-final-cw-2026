import numpy as np
import numpy.typing as npt
from typing import TypeAlias
import gensim.downloader as api
from pathlib import Path
import pandas as pd
import torch

# to run use python src/embeddings/pretrained/only_type_a_p_w2v.py in .venv312/Scripts/activate
Sentences: TypeAlias = list[str]
TokenisedSentences: TypeAlias = list[list[str]]
EmbeddingMatrix: TypeAlias = npt.NDArray[np.float32]


class PretrainedWord2VecEmbedder:
    def __init__(self, lowercase: bool = False):
        self.vector_size = 300
        self.lowercase = lowercase
        self.master_path = Path('src/data/type-a/master.csv')

        if not self.master_path.exists():
            raise FileNotFoundError(f'Master csv not found! {self.master_path}')

        print('Master csv found.')
        self.folder_path = Path('src/embeddings/computed-embeddings/type-a/p_w2v')
        self.df = pd.read_csv(self.master_path)

        print("PretrainedWord2VecEmbedder loading 'word2vec-google-news-300'")
        self.model = api.load('word2vec-google-news-300')
        print(f"PretrainedWord2VecEmbedder ready  vocab size: {len(self.model)}  dim: {self.vector_size}")

    def fit(self, _sentences: Sentences) -> 'PretrainedWord2VecEmbedder':
        return self
    
    def transform(self, sentence:str, idx:int) -> Path:
        tokens = sentence.lower().split() if self.lowercase else sentence.split()
        word_embeddings_found = []
        for t in tokens:
            if t not in self.model:
                continue 
            word_embeddings_found.append(self.model[t])
        if not word_embeddings_found:
            sentence_emb = np.zeros(self.vector_size, dtype=np.float32)
        else:
            sentence_emb = np.mean(word_embeddings_found, axis=0).astype(np.float32)
        sentence_tensor = torch.from_numpy(sentence_emb)
        filename = f'{idx}_p_wvec.pt'
        filepath = self.folder_path / filename
        torch.save(sentence_tensor, filepath)
        print(f'File: {filename}\nSaved to: {self.folder_path}')
        return filepath

    def oov_rate(self, sentences: Sentences) -> float:
        total = 0
        oov = 0

        for s in sentences:
            tokens = s.lower().split() if self.lowercase else s.split()
            total += len(tokens)
            oov += sum(1 for t in tokens if t not in self.model)

        rate = oov / total if total > 0 else 0.0
        print(f"OOV: {oov} / {total} tokens  ({rate * 100:.1f} %)")
        return rate

    def process(self):
        filepaths = []
        for idx, row in self.df.iterrows():
            sentence = row['label']
            filepaths.append(self.transform(sentence, idx))
        self.df['P_wvec'] = filepaths
        print(f'Success.{len(filepaths)} filepaths added')
        self.df.to_csv(self.master_path, index=False)