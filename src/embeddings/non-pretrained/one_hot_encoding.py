import numpy as np
import torch
from pathlib import Path
import pandas as pd
import sys 

class OneHot():
    """
    Big problem with one hot coding as embeddings are depenant on sentence length thus without fixed length, cant work in model
    Could try mean pooling np.array(embeddings).mean(axis=0) but will lose spatial info
    
    """
    def __init__(self):

        self.vocab = ['circle','triangle','square','diamond','hexagon',
                    'octagon', 'red', 'blue', 'green', 'yellow', 'orange',
                    'purple', 'pink', 'black','big', 'medium', 'small',
                    'above', 'below', 'left', 'of', 'right','a', 'is',
                    'the', 'positioned','can','be','seen']
        self.vocab_dict = {word: idx for idx, word in enumerate(self.vocab)}
        self.folder_path = Path('src/embeddings/computed-embeddings/type-a/OHC_emb')
        self._ROOT = Path(__file__).resolve().parent.parent.parent.parent
        self.master_path = Path('src/data/type-a/master.csv')
        self.df = pd.read_csv(self.master_path)

    def get_vocab(self):
        return self.vocab_dict
    
    def get_word_embedding(self, word):
        vector = np.zeros(len(self.vocab))
        vector[self.vocab_dict[word]] = 1
        return vector
    
    def get_sentence_embedding(self, sentence:str, idx:int):
        sentence_s = sentence.split()
        embeddings = []
        for word in sentence_s:
            if word not in self.vocab:
                raise ValueError('Word not in vocabulary')
            embeddings.append(self.get_word_embedding(word))
        embeddings = np.array(embeddings)
        print(embeddings)
        embeddings = torch.tensor(embeddings, dtype=torch.float)
        filename = f'{idx}.pt'
        filepath = self.folder_path / filename
        torch.save(embeddings, filepath)
        print(f'File: {filename}\nSaved to: {self.folder_path}')
        return filepath
    
    def process(self):
        filepaths = []
        for idx, row in self.df.iterrows():
            sentence = row['label']
            filepaths.append(self.get_sentence_embedding(sentence, idx))
        self.df['OneHot_emb'] = filepaths
        print(f'Success.{len(filepaths)} filepaths added')
        self.df.to_csv(self.master_path, index=False)


onehot = OneHot()

onehot.get_sentence_embedding('a red square', -1)
