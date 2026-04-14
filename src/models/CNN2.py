import torch.nn as nn

class CNN2(nn.Module):
    def __init__(self, output_dims:int):
        
        super().__init__()

        # Convolution + pooling layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), # Convolutional Layer 1
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # Convolutional Layer 2
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # Convolutional Layer 3
            nn.ReLU(),
            # Increase this for greater spatial awareness
            nn.AdaptiveAvgPool2d((3, 3)) 
        )

        self.fc_layers = nn.Sequential(
        # Defines one input layer and one output layer
    
            nn.Flatten(), # flattens image to long vector
            nn.Linear(128*3*3, 2500),
            nn.ReLU(),
            nn.Linear(2500, output_dims),

    )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
    