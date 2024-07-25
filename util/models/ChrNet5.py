import torch
from torch import nn
import torch.nn.functional as F

# CNN2 -> DNN
class Chromatin_Network5(nn.Module):
    def __init__(self, name, input_size=500):
        super(Chromatin_Network5, self,).__init__()
        self.name = name
        self.input_size = input_size

        if self.input_size == 500:
            self.layerinput = 500
            self.kerneldepth = 5
            self.paddingrow = 2
            self.paddingcol = 3
        if self.input_size == 800:
            self.layerinput = 800
            self.kerneldepth = 8
            self.paddingrow = 3
            self.paddingcol = 3
        if self.input_size == 33000:
            self.layerinput = 66400
            self.kerneldepth = 330
            self.paddingrow = 165
            self.paddingcol = 3

        self.conv1 = nn.Conv2d(1, 16, kernel_size=(self.kerneldepth,7), padding=(self.paddingrow,self.paddingcol))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(self.kerneldepth,7), padding=(self.paddingrow,self.paddingcol))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        
        self.fc1 = nn.Linear(self.layerinput, 64)
        self.fc2 = nn.Linear(64, 1)
        
    def forward(self, x):
        if self.input_size == 500:
            x = x.view(-1, 1, 5, 100)
        if self.input_size == 800:
            x = x.view(-1, 1, 8, 100)
        if self.input_size == 33000:
            x = x.view(-1, 1, 330, 100)

        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))

        return x
