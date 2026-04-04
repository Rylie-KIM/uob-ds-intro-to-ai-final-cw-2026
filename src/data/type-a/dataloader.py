from torch.utils.data import DataLoader
from type_a_dataloader import Dataset_A

dataset = Dataset_A('TB_pooler_emb')
dataset_loader = DataLoader(dataset, batch_size=1, shuffle=True)


for embedding, label in dataset_loader:
    print(f'This is the emb: {embedding}')
    print(f'Sentence: {label}')
    break
