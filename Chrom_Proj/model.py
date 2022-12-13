import torch
from torch import nn
import torch.nn.functional as F
import pdb
from torch.autograd import Variable 

"""
Neural Networks used in Study
"""

# DNN
class Chromatin_Network1(nn.Module):
    """
    Deep Neural Network
    """
    def __init__(self, name):
        super(Chromatin_Network1, self).__init__()
        self.name = name
        self.layer_1 = nn.Linear(500, 500) 
        self.layer_2 = nn.Linear(500, 500) 
        self.layer_3 = nn.Linear(500, 500) 
        self.layer_4 = nn.Linear(500, 500)
        self.layer_out = nn.Linear(500, 1) 


    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = F.relu(self.layer_3(x))
        x = F.relu(self.layer_4(x))
        x = torch.sigmoid(self.layer_out(x))
        
        return x

# CNN -> DNN
class Chromatin_Network2(nn.Module):
    """
    Convolutional Network to Deep Neural Network
    """
    def __init__(self, name):
        super(Chromatin_Network2,self).__init__()
        self.name = name
        self.CNNlayer_1 = nn.Conv1d(1, 3, 10) 
        self.CNNlayer_2 = nn.Conv1d(3, 5, 50) 
        self.CNNlayer_3 = nn.Conv1d(5, 10, 100)

        self.DNNlayer_0 = nn.Linear(3430, 500)
        self.DNNlayer_1 = nn.Linear(500, 500) 
        self.DNNlayer_2 = nn.Linear(500, 500) 
        self.DNNlayer_3 = nn.Linear(500, 500) 

        self.DNNlayer_4 = nn.Linear(500, 500)
        self.DNNlayer_5 = nn.Linear(500, 1)
    
        

    def forward(self,x):
        x = x.reshape(-1, 1, x.shape[1])
        x = F.relu(self.CNNlayer_1(x))
        x = F.relu(self.CNNlayer_2(x))
        x = F.relu(self.CNNlayer_3(x))

        x = torch.flatten(x, start_dim=1)

        x = F.relu(self.DNNlayer_0(x))
        x = F.relu(self.DNNlayer_1(x))
        x = F.relu(self.DNNlayer_2(x))
        x = F.relu(self.DNNlayer_3(x))
        x = F.relu(self.DNNlayer_4(x))
        x = torch.sigmoid(self.DNNlayer_5(x))
        
        return x

# LSTM -> DNN
class Chromatin_Network3(nn.Module):
    """
    Long Short Term Memory to Deep Neural Network
    """
    def __init__(self, name, hidden_size=30, num_layers=3):
        super(Chromatin_Network3, self).__init__()
        self.name = name
        self.num_layers = num_layers
        self.hidden_size = hidden_size


        self.lstm = nn.LSTM(input_size=500, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) #lstm

        self.DNNlayer_0 = nn.Linear(hidden_size, 500)
        self.DNNlayer_1 = nn.Linear(500, 500) 
        self.DNNlayer_2 = nn.Linear(500, 500) 
        self.DNNlayer_3 = nn.Linear(500, 500) 

        self.DNNlayer_4 = nn.Linear(500, 500)
        self.DNNlayer_5 = nn.Linear(500, 1)

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
        out = F.relu(self.DNNlayer_0(out))
        out = F.relu(self.DNNlayer_1(out))
        out = F.relu(self.DNNlayer_2(out))
        out = F.relu(self.DNNlayer_3(out))
        out = F.relu(self.DNNlayer_4(out))
        out = torch.sigmoid(self.DNNlayer_5(out))


        return out

# CNN -> LSTM -> DNN
class Chromatin_Network4(nn.Module):
    """
    Convolutional To LSTM To DNN
    """
    def __init__(self, name, hidden_size=1, num_layers=3):
        super(Chromatin_Network4, self).__init__()
        self.name = name
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        
        self.layer_1 = nn.Conv1d(1, 3, 10) 
        self.layer_2 = nn.Conv1d(3, 5, 50) 
        self.layer_3 = nn.Conv1d(5, 10, 100)

        self.lstm = nn.LSTM(input_size=3430, hidden_size=100,
                          num_layers=num_layers, batch_first=True) #lstm

        
        self.lin1 = nn.Linear(100,500)
        self.lin2 = nn.Linear(500,500)
        self.lin3 = nn.Linear(500,500)
        self.lin4 = nn.Linear(500,500)
        self.lin5 = nn.Linear(500,1)

        self.h_0 = None
        self.c_0 = None
        self.hidden = None

    def forward(self, x):
        x = x.reshape(-1, 1, x.shape[1])
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = F.relu(self.layer_3(x))
        
        x = torch.flatten(x, start_dim=1)
        
        if self.h_0 == None:    
            h_0 = Variable(torch.zeros(self.num_layers, 100)).to(x.device) #hidden state
            c_0 = Variable(torch.zeros(self.num_layers, 100)).to(x.device) #internal state
            self.hidden = (h_0, c_0)
        
        output, self.hidden = self.lstm(x, self.hidden) #lstm with input, hidden, and internal state

        out = F.relu(self.lin1(output))
        out = F.relu(self.lin2(out))
        out = F.relu(self.lin3(out))
        out = F.relu(self.lin4(out))
        out = torch.sigmoid(self.lin5(out))

        return out
