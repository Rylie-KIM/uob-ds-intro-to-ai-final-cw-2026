import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Convolution + pooling layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5), 
            nn.ReLU(),
            nn.MaxPool2d(2,2))   # conv1
        """
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(6, 16, kernel_size=5),  # conv2
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        """
        
    '''
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),  # fc1
            nn.ReLU(),
            nn.Linear(120, 84),         # fc2
            nn.ReLU(),
            nn.Linear(84, 10)           # fc3 (10 classes)
        )
    '''

    def forward(self, x):
        x = self.features(x)
        # x = self.classifier(x)
        print(f'Output:\n{x}')
        print(f'After Network shape: {x.shape}')
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
cnn = CNN()
print(f'Model Architecture:\n{cnn}')
cnn(data_t_n)

