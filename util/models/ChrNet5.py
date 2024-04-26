import torch
from torch import nn
import torch.nn.functional as F
from util.models import ChrNet1

class Chromatin_Network5(nn.Module):
    def __init__(self, name, input_size=500):
        super(Chromatin_Network5, self).__init__()
        self.name = name
        self.input_size = input_size
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),  # Add batch normalization
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),  # Add batch normalization
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Calculate layerinput based on input_size
        if self.input_size == 500:
            self.layerinput = 800
        elif self.input_size == 33000:
            self.layerinput = 65600
        else:
            raise ValueError("Invalid input_size. Supported values: 500, 33000")

        self.fc_layers = ChrNet1.Chromatin_Network1(name, self.layerinput, 1, 256)

    def forward(self, x):
        # Use adaptive pooling to handle inputs of different sizes
        x = F.adaptive_avg_pool2d(x, (100, 5 if self.input_size == 500 else 330))
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        x = torch.sigmoid(x)
        return x
