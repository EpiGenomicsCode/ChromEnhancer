import torch
from torch import nn

class Chromatin_Network(nn.Module):
  def __init__(self,input_shape):
    super(Chromatin_Network,self).__init__()
    self.fc1 = nn.Linear(input_shape,250)
    self.fc2 = nn.Linear(250,250)
    self.fc3 = nn.Linear(250,1)

  def forward(self,x):
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = torch.sigmoid(self.fc3(x))
    
    return x