import torch
import torch.nn as nn

# AlexNet adapted for 128×128 input that predicts a sentence embedding vector.

class AlexNet128(nn.Module):
    # embedding_dim: output size — must match the embedding model used for labels. (384 >> SBERT all-MiniLM-L6-v2, 768 >> BERT-base.)
    # dropout: dropout probability in the FC head.
    def __init__(self, embedding_dim: int = 384, dropout: float = 0.5):
        super().__init__()

        self.features = nn.Sequential(
            # block 1
            # Conv(64,11,s4,p2) >> BN >> ReLU >> MaxPool(3,s2) >> 15×15×64
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), #15×15×64

            # block 2
            # Conv(192,5,s1,p2) >> BN >> ReLU >> MaxPool(3,s2)   >>  7×7×192
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # 7×7×192

            # block 3
            # Conv(384,3,s1,p1) >> BN >> ReLU >>  7×7×384
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            # block 4
            #  Conv(256,3,s1,p1) >> BN >> ReLU >>  7×7×256
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # block 5
            #  Conv(256,3,s1,p1) >> BN >> ReLU >> MaxPool(3,s2)  >>  3×3×256
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), #3×3×256
        )

        # Keeps output 3×3 even if input slightly differs from 128×128
        # AdaptiveAvgPool(3×3)  >> Flatten >> 2304
        self.avgpool = nn.AdaptiveAvgPool2d((3, 3))         # 3×3×256 = 2304

        self.classifier = nn.Sequential(
            nn.Flatten(),
            #   FC(2304>>2048) >> ReLU >> Dropout
            nn.Linear(256 * 3 * 3, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            #   FC(2048>>1024) >> ReLU >> Dropout
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(1024, embedding_dim), 

        )

    # FC(1024>>embedding_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x
