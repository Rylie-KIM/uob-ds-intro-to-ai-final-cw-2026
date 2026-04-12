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