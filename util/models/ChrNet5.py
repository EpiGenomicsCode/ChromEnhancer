import torch
from torch import nn
import torch.nn.functional as F

# CNN2 -> DNN
class Chromatin_Network5(nn.Module):
    def __init__(self, name, input_size=500):
        super(Chromatin_Network5, self,).__init__()
        self.name = name
        self.input_size = input_size
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3,3), padding=(1,1))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3,3), padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        
        if self.input_size==500:
            self.layerinput = 800

        if self.input_size==4000:
            self.layerinput = 8000

        if self.input_size == 400:
            self.layerinput = 800

        if self.input_size == 32900:
            self.layerinput = 526336

        
        self.fc1 = nn.Linear(self.layerinput, 64)
        self.fc2 = nn.Linear(64, 1)
        
    def forward(self, x):
        if self.input_size==400:
            x = x.view(-1, 1, 100, 4)

        if self.input_size==500:
            x = x.view(-1, 1, 100, 5)

        if self.input_size==4000:
            x = x.view(-1, 1, 1000, 4)

        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))

        return x
