"""
src/models/cnn_1layer.py
1-block CNN: 128×128 RGB → sentence embedding.

Architecture (input 128×128×3):
  Conv(64,3,p1) → BN → ReLU → MaxPool(2,2)  →  64×64×64
  AdaptiveAvgPool(8×8) → Flatten → 4096
  FC(4096→512) → ReLU → Dropout
  FC(512→embedding_dim)

Note: output size 8×8 chosen so that 64÷8=8 divides evenly —
required for MPS (Apple Silicon) compatibility with AdaptiveAvgPool2d.
"""

import torch
import torch.nn as nn


class CNN1Layer(nn.Module):
    """
    Single-block CNN that predicts a sentence embedding vector from a 128×128 image.

    Args:
        embedding_dim: output size — must match the embedding model used for labels.
                       e.g. 384 for SBERT all-MiniLM-L6-v2, 768 for BERT-base.
        dropout:       dropout probability in the FC head.
    """

    def __init__(self, embedding_dim: int = 384, dropout: float = 0.5):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                 # → 64×64×64
        )

        self.avgpool = nn.AdaptiveAvgPool2d((8, 8))     # → 8×8×64 = 4096  (64÷8 exact, MPS-safe)

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
