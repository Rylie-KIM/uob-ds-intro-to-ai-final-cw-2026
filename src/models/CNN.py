import torch
import torch.nn as nn

class CNNStride(nn.Module):
    def __init__(self, output_dims : int, dropout : int = 0.5):
        super().__init__()

        self.convolution = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size = 3, stride=2, padding=1), # Convolutional Layer 1
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(16, 32, kernel_size = 3, stride=2, padding=1), # Convolutional Layer 2
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size = 3, stride=2, padding=1), # Convolutional Layer 3
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size = 3, stride=2, padding=1), # Convolutional Layer 4
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            
            nn.AdaptiveAvgPool2d((3,3))
        )
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*3*3, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, output_dims)
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.convolution(x)
        x = self.linear(x)
        return x
    