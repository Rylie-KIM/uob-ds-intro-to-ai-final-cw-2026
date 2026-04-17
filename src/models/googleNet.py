import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import GoogLeNet_Weights

class GoogleNet(nn.Module):

    def __init__(self, output_dims:int, dropout : int = 0.5):
        super().__init__()
        # Inputs : [batch_size, 3,224,224]
        self.convolution = models.googlenet(weights=GoogLeNet_Weights.DEFAULT)
        self.convolution.fc = nn.Identity()
        if self.convolution.aux_logits == True:
            raise ValueError('Aux_Logits still active')

        self.custom_linear = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, output_dims)                
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.convolution(x)
        x = self.custom_linear(x)
        return x