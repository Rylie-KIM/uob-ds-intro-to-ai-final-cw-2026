import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


class CNN2(nn.Module):
    def __init__(self, output_shape:int):
        
        super().__init__()

        # Convolution + pooling layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), # Convolutional Layer 1
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # Convolutional Layer 2
            nn.ReLU(),
            
            nn.Conv2d(64, 1, kernel_size = 1), # Convolutional Layer 3
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        ) # Output shape: [1, 262, 262]

        self.fc_layers = nn.Sequential(
        # Defines one input layer and one output layer
    
            nn.Flatten(), # flattens image to long vector
            # 68644 was chosen as thats the shape of the output after the conv layers.
            # This value needs to be configured to the number of cov layers
            nn.Linear(68644, 5000),
            nn.ReLU(),
            nn.Linear(5000, output_shape),

    )

    def forward(self, x):
        x = self.conv_layers(x)
        print(f'Output:\n{x}')
        print(f'After Conv shape: {x.shape}')
        x = self.fc_layers(x)
        print(f'Output Shape: {x.shape}')
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

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
path = r'C:\Users\fergu\Documents\GitHub\uob-ds-intro-to-ai-final-cw-2026\src\data_generation\type-a\type-a-dataset-png\1.png'
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
cnn = CNN2(output_shape = 768)
print(f'Model Architecture:\n{cnn}')
cnn(data_t_n)

# Hello world
# Hello again