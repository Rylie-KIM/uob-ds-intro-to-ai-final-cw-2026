from transformers import AutoModel, AutoTokenizer
import torch
from pathlib import Path
import pandas as pd
# TinyBERT (huawei-noah/TinyBERT_General_4L_312D) — pooler output ([CLS] token projection).
# Output dimension: 312

# @NOTE: do_lower_case=True is required — TinyBERT tokenizer does not recognise capitalised words.
class TinyBertPoolerEmbedder:
    def __init__(self):
        # TinyBert tokenizer doesnt recognise capitalised words
        self.model     = AutoModel.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')
        self.tokenizer = AutoTokenizer.from_pretrained('huawei-noah/TinyBERT_General_4L_312D', do_lower_case=True)
        self.master_path = Path('src/data/type-a/master.csv')
        self.root = Path('src/embeddings/computed-embeddings/type-a/Pooler_TB')
        self.df = pd.read_csv(self.master_path)


    def get_embedding(self, sentence: str, idx:int):
        """
        Sentence = 'hello my name is john'
        Tokenizer object converts sentence into dictionary of 'input_ids','token_type_ids','attention_mask' in tensor format.
        Converting to tensor now prevents need to do unsqueeze(0) to configure dimensiona
        No need for padding as each sentence is input to model individually with ouput fixed to (1,312)
        """
        processed = self.tokenizer(sentence, return_tensors='pt')
        with torch.no_grad():
            output = self.model(input_ids = processed['input_ids'], attention_mask = processed['attention_mask'])
        pooler = output.pooler_output
        filename = f'{idx}.pt'
        filepath = self.root / filename
        torch.save(pooler, filepath)
        print(f'File: {filename}\nSaved to: {self.root}')
        return filepath
    
    def process(self):
        filepaths = []
        for idx, row in self.df.iterrows():
            sentence = row['label']
            filepaths.append(self.get_embedding(sentence, idx))
        self.df['TB_Pooler_emb'] = filepaths
        print(f'Success.{len(filepaths)} filepaths added')
        self.df.to_csv(self.master_path, index=False)
    