
from transformers import AutoModel, AutoTokenizer, BertModel, BertTokenizer
import torch
import pandas as pd

#############
### Models ## 
#############
"""
Bert Model has output dimensions: (1, 768) (for each token embedding)
Tiny Bert Model has output dimensions: (1, 312) (for each token embedding)
"""
# TinyBert tokenizer doesnt recognise capitalised words
tinyBert_tokenizer = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D", do_lower_case=True)
tinyBert_model = AutoModel.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#############
### Paths ##
#############

sentence_dataset_path = r'Path_of_labelled_image_dataset'
# SUCH AS: sentence_dataset_path = r'C:\Users\fergu\Documents\GitHub\uob-ds-intro-to-ai-final-cw-2026\src\data_generation\type-a\type-a-dataset\as_eps.csv'
save_path = r'C:\Where_I_Want_It_To_Go'
file_name = 'embeddings_dataset'

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

def get_embeddings(model:object,ids:list):
    ########################################
    # Initial attempt to retrive embeddings#
    ########################################
    embeddings_full = []
    ids_t = torch.tensor([ids])
    with torch.no_grad():
        output = model(input_ids = ids_t)
    embeddings = output['last_hidden_state'][0]
    for i in range(len(ids)):
        embed = embeddings[i]
        embeddings_full.append(embed.detach().numpy())
    return embeddings_full

def get_pooler_embeddings(model:object,tokenizer:object, sentence:str):
    """
    Sentence = 'hello my name is john'
    Tokenizer object converts sentence into dictionary of 'input_ids','token_type_ids','attention_mask' in tensor format.
    Converting to tensor now prevents need to do unsqueeze(0) to configure dimensiona
    No need for padding as each sentence is input to model individually with ouput fixed to (1,768)
    """
    processed = tokenizer(sentence, return_tensors='pt')
    with torch.no_grad():
        output = model(input_ids = processed['input_ids'], attention_mask = processed['attention_mask'])
    return output[1] #.squeeze(0).tolist()

def get_output_embeddings(model,tokenizer, sentence):
    processed = tokenizer(sentence, return_tensors='pt')
    with torch.no_grad():
        output = model(input_ids = processed['input_ids'], attention_mask = processed['attention_mask'])
    return output[0] #.squeeze(0).tolist()

def get_mean_embeddings(model:object, tokenizer:object, sentence:str):
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

def process(sentences_file_path:str, model:object,tokenizer:object, save_path:str, file_name:str, embedding_type:str, embedding_dict):
    embedding = embedding_dict[embedding_type]
    df = pd.read_csv(sentences_file_path)
    df[embedding_type] = df['label'].apply(lambda sentence: embedding(model, tokenizer, sentence))
    csv_path = f'{save_path}_{file_name}.csv'
    df.to_csv(csv_path)

###########################
### Embedding Dictionary ##
###########################

embedding_dict = {
    'pooler_embeddings':get_pooler_embeddings,
    'output_embeddings':get_output_embeddings,
    'mean_embeddings':get_mean_embeddings
}

# process(sentence_dataset_path, model, tokenizer, save_path, file_name, 'mean_embeddings')
