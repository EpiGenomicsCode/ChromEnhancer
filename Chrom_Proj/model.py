import torch
from torch import nn
import torch.nn.functional as F
import pdb
from torch.autograd import Variable 


"""
Different Neural Networks
"""
# Convolutional Neural Network
class Chromatin_Network1(nn.Module):
    def __init__(self, name):
        super(Chromatin_Network1,self).__init__()
        self.name = name
        self.layer_1 = nn.Conv1d(1, 3, 10) 
        self.layer_2 = nn.Conv1d(3, 5, 50) 
        self.layer_3 = nn.Conv1d(5, 10, 100)

        self.layer_4 = nn.Linear(3430, 256)
        self.layer_5 = nn.Linear(256,128)
        self.layer_out = nn.Linear(128,1)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()

    def forward(self,x):
        x = x.reshape(-1, 1, x.shape[1])
        x = self.relu1(self.layer_1(x))
        x = self.relu2(self.layer_2(x))
        x = self.relu3(self.layer_3(x))

        x = torch.flatten(x, start_dim=1)

        x = self.relu4(self.layer_4(x))
        x = self.relu5(self.layer_5(x))
        x = torch.sigmoid(self.layer_out(x))

        return x

# Deep Neural Network
class Chromatin_Network2(nn.Module):
    def __init__(self, name):
        super(Chromatin_Network2, self).__init__()
        self.name = name
        self.layer_1 = nn.Linear(500, 250) 
        self.layer_2 = nn.Linear(250, 125) 
        self.layer_3 = nn.Linear(125, 62) 

        self.layer_4 = nn.Linear(62, 31)
        self.layer_5 = nn.Linear(31, 15)
        self.layer_out = nn.Linear(15, 1) 
        
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.layer_1(x))
        x = self.relu2(self.layer_2(x))
        x = self.relu3(self.layer_3(x))
        x = self.relu4(self.layer_4(x))
        x = self.relu5(self.layer_5(x))
        x = torch.sigmoid(self.layer_out(x))
        
        return x

# LSTM
class Chromatin_Network3(nn.Module):
    def __init__(self, name, input_size=1, hidden_size=30, num_layers=3):
        super(Chromatin_Network3, self).__init__()
        self.name = name
        self.num_layers = num_layers
        self.hidden_size = hidden_size


        self.lstm = nn.LSTM(input_size=500, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) #lstm

        self.lin1 = nn.Linear(hidden_size,25)
        self.lin2 = nn.Linear(25,10)
        self.lin3 = nn.Linear(10,5)
        self.lin4 = nn.Linear(5,1)

        self.h_0 = None
        self.c_0 = None
        self.hidden = None

    def forward(self, x):

        if self.h_0 == None:    
            h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(x.device) #hidden state
            c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(x.device) #internal state
            self.hidden = (h_0, c_0)

        x = x.reshape(-1,1,500)

        output, self.hidden = self.lstm(x, self.hidden) #lstm with input, hidden, and internal state

        out = output.reshape(-1,self.hidden_size)
        out = F.relu(self.lin1(out))
        out = F.relu(self.lin2(out))
        out = F.relu(self.lin3(out))
        out = torch.sigmoid(self.lin4(out))


        return out

