from transformers import AutoModel, AutoTokenizer
import torch


# @NOTE: do_lower_case=True is required — TinyBERT tokenizer does not recognise capitalised words.
class TinyBertPoolerEmbedder:
    def __init__(self):
        # TinyBert tokenizer doesnt recognise capitalised words
        self.model     = AutoModel.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')
        self.tokenizer = AutoTokenizer.from_pretrained('huawei-noah/TinyBERT_General_4L_312D', do_lower_case=True)

    def get_embedding(self, sentence: str) -> torch.Tensor:
        processed = self.tokenizer(sentence, return_tensors='pt')
        with torch.no_grad():
            output = self.model(input_ids = processed['input_ids'], attention_mask = processed['attention_mask'])
        pooler = output.pooler_output.squeeze(0)
        # print(f'P Shape: {pooler.shape}')
        filename = f'{idx}.pt'
        filepath = self.folder_path / filename
        torch.save(pooler, filepath)
        print(f'File: {filename}\nSaved to: {self.folder_path}')
        return filepath
    
    def process(self):
        filepaths = []
        for idx, row in self.df.iterrows():
            sentence = row['label']
            filepaths.append(self.get_embedding(sentence, idx))
        self.df['TB_pooler_emb'] = filepaths
        print(f'Success.{len(filepaths)} filepaths added')
        self.df.to_csv(self.master_path, index=False)
    
