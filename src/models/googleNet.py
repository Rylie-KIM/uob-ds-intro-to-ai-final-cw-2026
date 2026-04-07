import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import GoogLeNet_Weights

class GoogleNet(nn.Module):

    def __init__(self, output_dim):
        super().__init__()
        # Inputs : [batch_size, 3,224,224]
        self.encoder = models.googlenet(weights=GoogLeNet_Weights.DEFAULT)
        # Uses identiy to deactivate fc layer
        self.encoder.fc = nn.Identity()
        if self.encoder.aux_logits == True:
            raise ValueError('Aux_Logits still active')

        self.custom_fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)                
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.custom_fc(x)
        print(f'Output Shape:\n{x.shape}')
        print(f'Output:\n{x}')
        return x



from PIL import Image
path = r'C:\Users\fergu\Documents\GitHub\uob-ds-intro-to-ai-final-cw-2026\src\data\images\type-a\a_1.png'
img = Image.open(path).convert('RGB')

transforms = GoogLeNet_Weights.DEFAULT.transforms()

img = transforms(img)

print(img.shape)

cnn = GoogleNet(output_dim = 512)
print(f'Input Shape: {img.unsqueeze(0).shape}')
cnn(img.unsqueeze(0))