from transformers import AutoModel, AutoTokenizer
import torch
#TinyBERT (huawei-noah/TinyBERT_General_4L_312D) — mean pooling over all tokens.
# Output dimension: 312

# @NOTE: do_lower_case=True is required — TinyBERT tokenizer does not recognise capitalised words.
class TinyBertMeanEmbedder:
    def __init__(self):
        # TinyBert tokenizer doesnt recognise capitalised words
        self.model     = AutoModel.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')
        self.tokenizer = AutoTokenizer.from_pretrained('huawei-noah/TinyBERT_General_4L_312D', do_lower_case=True)

    def get_embedding(self, sentence: str) -> torch.Tensor:
        """
        Get mean embeddings function calculates the mean embedding from the embedding tensors produced in the model.
        Output : torch.Size([312])

        NOTE:
        - The method INCLUDES CLS and SEP tokens in mean calculations and DOES NOT adjust for padding
        - To adjust for padding, multiply attention mask vector by token embeddings where attention mask is
        either 1 or 0.
        - To adjust for CLS and SEP tokens, remove first and last token in output.last_hidden_state

        """
        processed = self.tokenizer(sentence, return_tensors='pt')
        with torch.no_grad():
            output = self.model(input_ids = processed['input_ids'], attention_mask = processed['attention_mask'])
        last_hidden_state = output['last_hidden_state'][0]
        # print(f'LH Shape: {last_hidden_state.shape}')
        last_hidden_state = last_hidden_state[1:-1]
        mean_emb = last_hidden_state.mean(dim=0)
        # print(f'Output Shape: {mean_emb.shape}')
        return mean_emb
    
tb_emb = TinyBertMeanEmbedder()

print(tb_emb.get_embedding('hello my name is bob'))