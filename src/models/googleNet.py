import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import GoogLeNet_Weights

class GoogleNet(nn.Module):

    def __init__(self, output_dims):
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
            nn.Linear(512, output_dims)                
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.custom_fc(x)
        return x