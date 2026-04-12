from torch.utils.data import DataLoader
from type_a_dataloader import Dataset_A
import torchvision.transforms as transforms

transform = transforms.Compose([
    # May want to vary the resizing and see how that effects the accuracy
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

"""
For googlenet model:
weights = GoogLeNet_Weights.DEFAULT
transform = weights.transforms()

"""
embedding_types = ['TB_pooler_emb','TB_mean_emb','B_pooler_emb','B_mean_emb','sbert_emb']

for embedding_type in embedding_types:
    dataset = Dataset_A(embedding_type, transform_imgs = transform)
    data_loader = DataLoader(dataset, batch_size = 32, shuffle=True)
    for img_embedding, sentence_emb in data_loader:
        print(f'DATASET: {embedding_type}\nImage_emb shape: {img_embedding.shape}\nSentence_emb shape: {sentence_emb.shape}')
        print('------------------------------------------')
        break