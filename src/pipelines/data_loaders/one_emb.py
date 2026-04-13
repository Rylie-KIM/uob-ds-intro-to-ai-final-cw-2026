import pandas as pd
import torch
from pathlib import Path
import torchvision.transforms as transforms
from PIL import Image
from natsort import natsorted

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

emb_dict_paths = {
    'TB_pooler_emb': Path('C:/Masters/Text Analytics/AI_Coursework/embeddings/Pooler_TB'),
    'TB_mean_emb': Path('C:/Masters/Text Analytics/AI_Coursework/embeddings/Mean_TB'),
    'B_pooler_emb': Path('C:/Masters/Text Analytics/AI_Coursework/embeddings/Pooler_B'),
    'B_mean_emb': Path('C:/Masters/Text Analytics/AI_Coursework/embeddings/Mean_B'),
    'sbert_emb': Path('C:/Masters/Text Analytics/AI_Coursework/embeddings/sbert')
}
emb_dict_output = {
            'TB_pooler_emb':Path("C:/Masters/Text Analytics/AI_Coursework/embeddings/TB_pooler_emb_master.pt"),
            'TB_mean_emb': Path("C:/Masters/Text Analytics/AI_Coursework/embeddings/TB_mean_emb_master.pt"),
            'B_pooler_emb': Path("C:/Masters/Text Analytics/AI_Coursework/embeddings/B_pooler_emb_master.pt"),
            'B_mean_emb': Path("C:/Masters/Text Analytics/AI_Coursework/embeddings/B_mean_emb_master.pt"),
            'sbert_emb': Path("C:/Masters/Text Analytics/AI_Coursework/embeddings/sbert_emb_master.pt")
        }

def combine_sentence_embs(emb_type:str, emb_dict_paths:dict, emb_dict_output:dict):
    emb_path = emb_dict_paths[emb_type]
    output_path = emb_dict_output[emb_type]
    files = emb_path.glob('*.pt')
    ordered_files = natsorted(files)
    print(f'{len(ordered_files)} files found for {emb_type}')
    all_embs = []
    for file in ordered_files:
        emb = torch.load(file)
        all_embs.append(emb)
    print(f'Appended files for {emb_type}')
    output = torch.stack(all_embs)
    torch.save(output, output_path)
    print(f'Saved {output.shape} in {output_path}')

def convert_combine_images(transform, file_dir_path, output_path):
    files = file_dir_path.glob('*.png')
    ordered_files = natsorted(files)
    print(f'{len(ordered_files)} image files found')
    all_embs = []
    for file in ordered_files:
        img = Image.open(file).convert('RGB')
        img_emb = transform(img)
        all_embs.append(img_emb)
    print(f'Stacking files...')
    output = torch.stack(all_embs)
    torch.save(output, output_path)
    print(f'Saved {output.shape} in {output_path}')

sentence_emb_types = [
            'TB_pooler_emb',
            'TB_mean_emb',
            'B_pooler_emb',
            'B_mean_emb',
            'sbert_emb',
            ]

for emb in sentence_emb_types:
    combine_sentence_embs(emb, emb_dict_paths, emb_dict_output)

img_path = Path('C:/Masters/Text Analytics/AI_Coursework/images/png')
output_path = Path("C:/Masters/Text Analytics/AI_Coursework/embeddings/images_master.pt")
convert_combine_images(transform, img_path, output_path)
