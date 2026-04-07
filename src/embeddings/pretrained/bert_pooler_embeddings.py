from transformers import BertModel, BertTokenizer
import torch

# BERT (bert-base-uncased) — pooler output ([CLS] token projection).
# Output dimension: 768
class BertPoolerEmbedder:
    def __init__(self):
        self.model     = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def get_embedding(self, sentence: str) -> torch.Tensor:
        """
        Sentence = 'hello my name is john'
        Tokenizer object converts sentence into dictionary of 'input_ids','token_type_ids','attention_mask' in tensor format.
        Converting to tensor now prevents need to do unsqueeze(0) to configure dimensiona
        No need for padding as each sentence is input to model individually with ouput fixed to (1,768)
        """

        processed = self.tokenizer(sentence, return_tensors='pt')
        with torch.no_grad():
            output = self.model(input_ids = processed['input_ids'], attention_mask = processed['attention_mask'])
        return output.pooler_output.squeeze(0)