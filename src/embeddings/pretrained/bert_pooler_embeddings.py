from transformers import BertModel, BertTokenizer
import torch

# BERT (bert-base-uncased) — pooler output ([CLS] token projection).
# Output dimension: 768
class BertPoolerEmbedder:
    def __init__(self):
        self.model     = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def get_embedding(self, sentence: str) -> torch.Tensor:
        processed = self.tokenizer(sentence, return_tensors='pt')
        with torch.no_grad():
            output = self.model(input_ids = processed['input_ids'], attention_mask = processed['attention_mask'])
        return output.pooler_output.squeeze(0)