"""
src/models/cnn_2layer.py
2-block CNN: 128×128 RGB → sentence embedding.

Architecture (input 128×128×3):
  Conv(32,3,p1) → BN → ReLU → MaxPool(2,2)  →  64×64×32
  Conv(64,3,p1) → BN → ReLU → MaxPool(2,2)  →  32×32×64
  AdaptiveAvgPool(4×4) → Flatten → 1024
  FC(1024→512) → ReLU → Dropout
  FC(512→embedding_dim)

Note: output size 4×4 chosen so that 32÷4=8 divides evenly —
required for MPS (Apple Silicon) compatibility with AdaptiveAvgPool2d.
"""

import torch
import torch.nn as nn


class CNN2Layer(nn.Module):
    """
    Two-block CNN that predicts a sentence embedding vector from a 128×128 image.

    Args:
        embedding_dim: output size — must match the embedding model used for labels.
                       e.g. 384 for SBERT all-MiniLM-L6-v2, 768 for BERT-base.
        dropout:       dropout probability in the FC head.
    """

    def __init__(self, embedding_dim: int = 384, dropout: float = 0.5):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3,  32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                 # → 64×64×32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                 # → 32×32×64
        )

        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))     # → 4×4×64 = 1024  (32÷4 exact, MPS-safe)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x
