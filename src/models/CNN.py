import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms


class CNN_encoder(nn.Module):
    def __init__(self, embedding_dims):
        super().__init__()
        # Input: 3, 512, 512
        # Convolution + pooling layers
        self.convolution = nn.Sequential(
            # Preferred the use of strided convolutions over max pooling
            nn.Conv2d(3, 16, 3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), 
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1,1))
        )
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, embedding_dims)
        )

    def forward(self, x):
        x = self.convolution(x)
        x = self.linear(x)
        return x
    