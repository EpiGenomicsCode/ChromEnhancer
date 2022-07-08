import torch
from torch import nn
import torch.nn.functional as F

class Chromatin_Network(nn.Module):
  def __init__(self,input_shape):
    super(Chromatin_Network,self).__init__()
    self.conv1 = nn.Conv2d(1, 3, 2) 
    self.conv2 = nn.Conv2d(3, 6, 3)

    self.fc1 = nn.Linear(6*7*7, 256)
    self.fc2 = nn.Linear(256,128)
    self.fc3 = nn.Linear(128,1)

  def forward(self,x):
    x = x.reshape(-1,1,10,10)

    x = torch.relu(self.conv1(x))
    x = torch.relu(self.conv2(x))

    x = torch.flatten(x, start_dim=1)
    
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = torch.sigmoid(self.fc3(x))

    return torch.round(x)