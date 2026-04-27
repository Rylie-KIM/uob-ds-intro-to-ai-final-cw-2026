from transformers import BertModel, BertTokenizer
import torch
from pathlib import Path
import pandas as pd

# BERT (bert-base-uncased) — mean pooling over all tokens.
# Output dimension: 768
# Output emb: torch.Size([768])
class BertMeanEmbedder:
    def __init__(self):
        self.model     = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.master_path = Path('src/data/type-a/master.csv')
        self.folder_path = Path('src/embeddings/computed-embeddings/type-a/Mean_B')
        self.folder_path.mkdir(parents=True, exist_ok=True)
        self.df = pd.read_csv(self.master_path)
        self.model.eval()


    def get_embedding(self, sentence: str, idx:int):

        processed = self.tokenizer(sentence, return_tensors='pt')
        with torch.no_grad():
            output = self.model(input_ids = processed['input_ids'], attention_mask = processed['attention_mask'])
        last_hidden_state = output['last_hidden_state'][0]
        last_hidden_state = last_hidden_state[1:-1]
        print(f'LHS Size: {last_hidden_state.shape}')
        mean_emb = last_hidden_state.mean(dim=0)
        print(f'Output Shape: {mean_emb.shape}')
        filename = f'{idx}_B_mean.pt'
        filepath = self.folder_path / filename
        torch.save(mean_emb, filepath)
        print(f'File: {filename}\nSaved to: {self.folder_path}')
        return filepath
    
    def process(self):
        filepaths = []
        for idx, row in self.df.iterrows():
            sentence = row['label']
            filepaths.append(self.get_embedding(sentence, idx))
        self.df['B_mean_emb'] = filepaths
        print(f'Success.{len(filepaths)} filepaths added')
        self.df.to_csv(self.master_path, index=False)
