import torch
from torch import nn
import torch.nn.functional as F

#inputsize = 1
#seq_len=500
# numlayers = 2

class Chromatin_Network1(nn.Module):
    def __init__(self, name):
        super(Chromatin_Network1, self).__init__()
        self.name = name
        self.layer_1 = nn.Linear(500, 64) 
        self.layer_2 = nn.Linear(64, 64)
        self.layer_out = nn.Linear(64, 1) 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = torch.sigmoid(self.layer_out(x))

        
        return x


class Chromatin_Network2(nn.Module):
    def __init__(self, name):
        super(Chromatin_Network2, self).__init__()
        self.name = name
        self.layer_1 = nn.Linear(500, 64) 
        self.layer_2 = nn.Linear(64, 64) 
        self.layer_3 = nn.Linear(64, 64) 
        self.layer_4 = nn.Linear(64, 64)
        self.layer_out = nn.Linear(64, 1) 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.relu(self.layer_3(x))
        x = self.relu(self.layer_4(x))
        x = torch.sigmoid(self.layer_out(x))

        
        return x