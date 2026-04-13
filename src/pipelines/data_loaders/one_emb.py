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

resize = transforms.Resize((128,128))
_ROOT = Path(__file__).resolve().parent.parent.parent.parent

emb_dict_paths = {
    'TB_pooler_emb': Path('C:/Masters/Text Analytics/AI_Coursework/embeddings/Pooler_TB'),
    'TB_mean_emb': Path('C:/Masters/Text Analytics/AI_Coursework/embeddings/Mean_TB'),
    'B_pooler_emb': Path('C:/Masters/Text Analytics/AI_Coursework/embeddings/Pooler_B'),
    'B_mean_emb': Path('C:/Masters/Text Analytics/AI_Coursework/embeddings/Mean_B'),
    'sbert_emb': Path('C:/Masters/Text Analytics/AI_Coursework/embeddings/sbert')
}

emb_dict_output = {
        
            'TB_pooler_emb': _ROOT / 'src' / 'embeddings' / 'computed-embeddings'/ 'type-a' / 'results' / 'TB_pooler_emb_master.pt',
            'TB_mean_emb': _ROOT / 'src' / 'embeddings' / 'computed-embeddings'/ 'type-a' / 'results' / 'TB_mean_emb_master.pt',
            'B_pooler_emb': _ROOT / 'src' / 'embeddings' / 'computed-embeddings' / 'type-a' / 'results' / 'B_pooler_emb_master.pt',
            'B_mean_emb': _ROOT / 'src' / 'embeddings' / 'computed-embeddings'/ 'type-a' / 'results' / 'B_mean_emb_master.pt',
            'sbert_emb': _ROOT / 'src' / 'embeddings' / 'computed-embeddings'/ 'type-a' / 'results' / 'sbert_emb_master.pt'
        }

def combine_sentence_embs(emb_dict_paths:dict, emb_dict_output:dict, emb_type:str):
    emb_path = emb_dict_paths[emb_type]
    output_path = emb_dict_output[emb_type]
    files = emb_path.glob('*.pt')
    ordered_files = natsorted(files)
    print(f'{len(ordered_files)} files found for {emb_type}')
    all_embs = []
    for file in ordered_files:
        emb = torch.load(file, weights_only=True)
        all_embs.append(emb)
    print(f'Appended files for {emb_type}')
    output = torch.stack(all_embs)
    torch.save(output, output_path)
    print(f'Saved {output.shape} in {output_path}')

def convert_combine_images(transform, image_path, output_path):
    files = image_path.glob('*.png')
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
master_path = r'C:\Users\fergu\Documents\GitHub\uob-ds-intro-to-ai-final-cw-2026\src\data\type-a\master.csv'
df = pd.read_csv(master_path)

# Check first few CSV rows
print(df[['path', 'TB_pooler_emb']].head())

# Check first few files in each folder
img_files = natsorted(Path('C:/Masters/Text Analytics/AI_Coursework/images/png').glob('*.png'))
emb_files = natsorted(Path('C:/Masters/Text Analytics/AI_Coursework/embeddings/Pooler_TB').glob('*.pt'))

print([f.name for f in img_files[:5]])
print([f.name for f in emb_files[:5]])


for emb in sentence_emb_types:
    combine_sentence_embs(emb_dict_paths,emb_dict_output, emb)

img_path = Path('C:/Masters/Text Analytics/AI_Coursework/images/png')
output_path = _ROOT / 'src' / 'embeddings' / 'computed-embeddings' / 'type-a' / 'results' /'images_master.pt'

convert_combine_images(transform, img_path, output_path)
