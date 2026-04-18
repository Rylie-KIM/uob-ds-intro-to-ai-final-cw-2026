import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

class ResnetDSA(nn.Module):

    def __init__(self, output_dims: int, dropout: float = 0.5):
        super().__init__()

        self.convolution = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.convolution.fc = nn.Identity()
        self.custom_linear = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, output_dims)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convolution(x)
        x = self.custom_linear(x)
        return x
