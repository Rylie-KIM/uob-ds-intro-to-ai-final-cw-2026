
from transformers import AutoModel, AutoTokenizer, BertModel, BertTokenizer
import torch
import pandas as pd


"""
Bert Model has output dimensions: (1, 768) (for each token embedding)
Tiny Bert Model has output dimensions: (1, 312) (for each token embedding)
"""



##########################
### Embedding Functions ##
##########################

"""

Visually, output.last_hidden_state (batches ie sentences, num_tokens, feature_dimenions) :

            tensor = [ 
            
            [   [ 0.21, 0.65, 0.38....0.54 ] Token 1
                [ 0.25, 0.75, 0.18....0.68 ] Token 2
            ....
                [ z, y, x ] ] Token n
            ]
Therefore we use dim=1
"""

class BertEmbeddings:
    """
    PATHS: 

    sentence_dataset_path = r'Path_of_labelled_image_dataset.csv'
    save_path = r'C:\Where_I_Want_It_To_Go'
    file_name = 'embeddings_dataset'
    
    
    """
    def __init__(self, sentence_dataset_path:str, save_path:str, file_name:str):
        
        self.embedding_dict = {
            'pooler_embeddings':self.get_pooler_embeddings,
            'output_embeddings':self.get_output_embeddings,
            'mean_embeddings':self.get_mean_embeddings
        }
        self.sentence_dataset_path = sentence_dataset_path
        self.save_path = save_path
        self.file_name = file_name

    #############
    ### Models ## 
    #############
        # TinyBert tokenizer doesnt recognise capitalised words
        self.tinyBert_tokenizer = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D", do_lower_case=True)
        self.tinyBert_model = AutoModel.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.model = {
            'bert': self.bert_model,
            'tiny_bert':self.tinyBert_model
        }
        self.tokenizer = {
            'bert_tokenizer':self.bert_tokenizer,
            'tinyBert_tokenizer':self.tinyBert_tokenizer
        }

    def get_pooler_embeddings(self, model:object,tokenizer:object, sentence:str):
        """
        Sentence = 'hello my name is john'
        Tokenizer object converts sentence into dictionary of 'input_ids','token_type_ids','attention_mask' in tensor format.
        Converting to tensor now prevents need to do unsqueeze(0) to configure dimensiona
        No need for padding as each sentence is input to model individually with ouput fixed to (1,768)
        """

        processed = tokenizer(sentence, return_tensors='pt')
        with torch.no_grad():
            output = model(input_ids = processed['input_ids'], attention_mask = processed['attention_mask'])
        return output.pooler_output #.squeeze(0).tolist()

    def get_output_embeddings(self, model:object,tokenizer:object, sentence:str):
        

        processed = tokenizer(sentence, return_tensors='pt')
        with torch.no_grad():
            output = model(input_ids = processed['input_ids'], attention_mask = processed['attention_mask'])
        return output.last_hidden_state #.squeeze(0).tolist()

    def get_mean_embeddings(self, model:object, tokenizer:object, sentence:str):
        """
        Get mean embeddings function calculates the mean embedding from the embedding tensors produced in the model.

        NOTE: 
        - The method INCLUDES CLS and SEP tokens in mean calculations and DOES NOT adjust for padding
        - To adjust for padding, multiply attention mask vector by token embeddings where attention mask is
        either 1 or 0. 
        - To adjust for CLS and SEP tokens, remove first and last token in output.last_hidden_state
        
        """
        processed = tokenizer(sentence, return_tensors='pt')
        with torch.no_grad():
            output = model(input_ids = processed['input_ids'], attention_mask = processed['attention_mask'])
        last_hidden_state = output['last_hidden_state']
        sum_emb = last_hidden_state.sum(dim=1)
        # Calculates num embeddings based off len(num_tokens)
        num_embeddings = len(processed['input_ids'].squeeze(0).tolist())
        mean_emb = sum_emb/num_embeddings
        return mean_emb #.squeeze(0).tolist()

    def process(self, model:str,tokenizer:str, embedding_type:str):
        
        model = self.model[model]
        tokenizer = self.tokenizer[tokenizer]
        embedding = self.embedding_dict[embedding_type]
        df = pd.read_csv(self.sentences_file_path)
        df[embedding_type] = df['label'].apply(lambda sentence: embedding(model, tokenizer, sentence))
        csv_path = f'{self.save_path}_{self.file_name}.csv'
        df.to_csv(csv_path)



# sentence_dataset_path = r''
# save_path = r''
# file_name = r''
# tiny_bert_embeddings = BertEmbeddings(sentence_dataset_path, save_path, file_name)
# tiny_bert_embeddings.process(model = 'tiny_bert',tokenizer = 'tinyBert_tokenizer', embedding_type = 'mean_embeddings' )

