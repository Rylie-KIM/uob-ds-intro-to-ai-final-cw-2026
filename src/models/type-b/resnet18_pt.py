import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


class ResNet18Pretrained(nn.Module):
    #  With 8,006 training samples and 11 M parameters, scratch training yields about 1,374 params per sample — high overfitting risk. 
    # ImageNet pretrained weights provide a well-generalised starting point; only the FC head (~200 K params) is initialised randomly, reducing the effective unconstrained parameter count.

    def __init__(self, embedding_dim: int = 384, dropout: float = 0.5):
        super().__init__()

        backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        # Replace the ImageNet classification head (512 >> 1000) with an embedding projection head (512 >> embedding_dim)
        backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(512, embedding_dim),
        )

        self.model = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
