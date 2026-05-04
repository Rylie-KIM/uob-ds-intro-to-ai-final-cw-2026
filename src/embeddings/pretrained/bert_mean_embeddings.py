from transformers import BertModel, BertTokenizer
import torch

# BERT (bert-base-uncased) — mean pooling over all tokens.
# Output dimension: 768
class BertMeanEmbedder:
    def __init__(self):
        self.model     = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def get_embedding(self, sentence: str) -> torch.Tensor:
        processed = self.tokenizer(sentence, return_tensors='pt')
        with torch.no_grad():
            output = self.model(input_ids = processed['input_ids'], attention_mask = processed['attention_mask'])
        last_hidden_state = output['last_hidden_state'][0]
        last_hidden_state = last_hidden_state[1:-1]
        print(f'LS Size: {last_hidden_state.shape}')
        mean_emb= last_hidden_state.mean(dim=0)
        print(f'Output Shape: {mean_emb.shape}')
        return mean_emb
    