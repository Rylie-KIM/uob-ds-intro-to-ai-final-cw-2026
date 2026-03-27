
from transformers import AutoModel, AutoTokenizer
import torch
import pandas as pd

tokenizer = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
model = AutoModel.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")

path = r'C:\Users\fergu\Documents\GitHub\uob-ds-intro-to-ai-final-cw-2026\src\data_generation\type-a\type-a-dataset\as_eps.csv'

def get_embeddings(model:object,ids:list):
    embeddings_full = []
    ids_t = torch.tensor([ids])
    with torch.no_grad():
        output = model(input_ids = ids_t)
    embeddings = output['last_hidden_state'][0]
    for i in range(len(ids)):
        embed = embeddings[i]
        embeddings_full.append(embed.detach.numpy())
    return embeddings_full

def process(sentences_file_path:str, model:object, save_path:str, file_name:str):
    tokenized_sentences = []
    df = pd.read_csv(sentences_file_path)
    df['tokenized'] = df['label'].apply(lambda x: tokenizer.tokenize(x))
    df['ids'] = df['tokenized'].apply(lambda x: tokenizer.convert_tokens_to_ids(x))
    df['embeddings'] = df['ids'].apply(lambda x: get_embeddings(model, x))
    csv_path = f'{save_path}_{file_name}.csv'
    df.to_csv(csv_path)

print(process(path, model))