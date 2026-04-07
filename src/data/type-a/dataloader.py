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


dataset = Dataset_A('TB_pooler_emb', transform_img = transform)
dataset_loader = DataLoader(dataset, batch_size=1, shuffle=True)


for embedding, label in dataset_loader:
    print(f'This is the emb: {embedding}')
    print(f'Sentence: {label}')
    break

