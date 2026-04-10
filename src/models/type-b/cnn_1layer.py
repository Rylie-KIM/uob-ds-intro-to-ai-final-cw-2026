import torch
import torch.nn as nn


class CNN1Layer(nn.Module):
    def __init__(self, embedding_dim: int = 384, dropout: float = 0.5):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                 #  64×64×64
        )

        self.avgpool = nn.AdaptiveAvgPool2d((8, 8))     #  8×8×64 = 4096  (64÷8 exact, MPS-safe)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x
