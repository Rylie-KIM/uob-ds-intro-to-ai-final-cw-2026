
from transformers import AutoModel, AutoTokenizer
import torch
import pandas as pd

tokenizer = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
model = AutoModel.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")

path = r'C:\Users\fergu\Documents\GitHub\uob-ds-intro-to-ai-final-cw-2026\src\data_generation\type-a\type-a-dataset\as_eps.csv'
def tokenize(sentences_file_path):
    tokenized_sentences = []
    df = pd.read_csv(path)
    df['tokenized'] = df['label'].apply(lambda x: tokenizer.tokenize(x))
    df['ids'] = df['tokenized'].apply(lambda x: tokenizer.convert_tokens_to_ids(x))
    df['embeddings'] = df['ids'].apply(lambda x: )





print(tokenize(path))