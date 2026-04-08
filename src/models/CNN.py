"""
src/models/CNN.py
Simple CNN adapted for 128×128 RGB input → sentence embedding output.

Architecture (input 128×128×3):
  Conv(32,3,p1) → BN → ReLU → MaxPool(2,2)   → 64×64×32
  Conv(64,3,p1) → BN → ReLU → MaxPool(2,2)   → 32×32×64
  Conv(128,3,p1)→ BN → ReLU → MaxPool(2,2)   → 16×16×128
  AdaptiveAvgPool(4×4) → Flatten → 2048
  FC(2048→512) → ReLU → Dropout
  FC(512→embedding_dim)
"""

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
        print(f'Pre Shape: {x.shape}')
        x = self.convolution(x)
        x = self.linear(x)
        print(f'Output:\n{x}')
        print(f'Output Shape:\n{x.shape}')
        return x
    

def train_model(model, train_loader, test_loader, epochs=20, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    dev_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        dev_running = 0.0

        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                dev_loss = criterion(outputs, labels)
                dev_running += dev_loss.item()

        avg_dev_loss = dev_running / len(test_loader)
        dev_losses.append(avg_dev_loss)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_dev_loss:.4f}")

    return train_losses, dev_losses

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

path = os.path.join(os.path.dirname(__file__), '..', 'data', 'images', 'type-a', 'a_1.png')
img = Image.open(path).convert('RGB')
# img.show()
np_img = np.array(img)
print(np_img.shape)
# plt.imshow(np_img)
# plt.axis('off')
# plt.show()


# Convert to tensor
converter = transforms.ToTensor()
data_t = converter(np_img)
# Normalise
normalise = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
data_t_n = normalise(data_t)
print(f'NEW: {data_t_n.shape}')
cnn = CNN_encoder(embedding_dims = 512)
print(f'Model Architecture:\n{cnn}')
cnn(data_t_n.unsqueeze(0))

