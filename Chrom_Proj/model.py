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
        self.layer_1 = nn.Linear(500, 250) 
        self.layer_2 = nn.Linear(250, 125) 
        self.layer_3 = nn.Linear(125, 62) 

        self.layer_4 = nn.Linear(62, 31)
        self.layer_5 = nn.Linear(31, 15)
        self.layer_6 = nn.Linear(15, 7)
        self.layer_7 = nn.Linear(7, 3)
        self.layer_out = nn.Linear(3, 1) 
        
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()
        self.relu6 = nn.ReLU()
        self.relu7 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.layer_1(x))
        x = self.relu2(self.layer_2(x))
        x = self.relu3(self.layer_3(x))
        x = self.relu4(self.layer_4(x))
        x = self.relu5(self.layer_5(x))
        x = self.relu6(self.layer_6(x))
        x = self.relu7(self.layer_7(x))
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
        self.layer_1 = nn.Conv1d(1, 3, 10) 
        self.layer_2 = nn.Conv1d(3, 5, 50) 
        self.layer_3 = nn.Conv1d(5, 10, 100)

        self.layer_4 = nn.Linear(3430, 500)

        self.DNN = Chromatin_Network1(self.name+"_DNN")
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        

    def forward(self,x):
        x = x.reshape(-1, 1, x.shape[1])
        x = self.relu1(self.layer_1(x))
        x = self.relu2(self.layer_2(x))
        x = self.relu3(self.layer_3(x))

        x = torch.flatten(x, start_dim=1)

        x = self.relu4(self.layer_4(x))
        x = self.DNN(x)

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

# CNN -> LSTM -> DNN
class Chromatin_Network4(nn.Module):
    """
    Convolutional To LSTM To DNN
    """
    def __init__(self, name, hidden_size=500, num_layers=5):
        super(Chromatin_Network4, self).__init__()
        self.name = name
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        
        self.layer_1 = nn.Conv1d(1, 3, 10) 
        self.layer_2 = nn.Conv1d(3, 5, 50) 
        self.layer_3 = nn.Conv1d(5, 10, 100)

        self.lstm = nn.LSTM(input_size=3430, hidden_size=100,
                          num_layers=num_layers, batch_first=True) #lstm

        
        self.lin1 = nn.Linear(100,50)
        self.lin2 = nn.Linear(50,20)
        self.lin3 = nn.Linear(20,10)
        self.lin4 = nn.Linear(10,1)

        self.h_0 = None
        self.c_0 = None
        self.hidden = None

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()
        self.relu6 = nn.ReLU()
        self.relu7 = nn.ReLU()

    def forward(self, x):
        x = x.reshape(-1, 1, x.shape[1])
        x = self.relu1(self.layer_1(x))
        x = self.relu2(self.layer_2(x))
        x = self.relu3(self.layer_3(x))
        
        x = torch.flatten(x, start_dim=1)
        
        if self.h_0 == None:    
            h_0 = Variable(torch.zeros(self.num_layers, 100)).to(x.device) #hidden state
            c_0 = Variable(torch.zeros(self.num_layers, 100)).to(x.device) #internal state
            self.hidden = (h_0, c_0)
        
        output, self.hidden = self.lstm(x, self.hidden) #lstm with input, hidden, and internal state

        out = self.relu4(self.lin1(output))
        out = self.relu5(self.lin2(out))
        out = self.relu6(self.lin3(out))
        out = self.relu7(self.lin4(out))
        
        out = torch.sigmoid(out)


        return out

# Probably dont need

# DNN -> LSTM
class Chromatin_Network5(nn.Module):
    """
    Deep Nerual Network to LSTM
    """
    def __init__(self, name, hidden_size=1, num_layers=3):
        super(Chromatin_Network5, self).__init__()
        self.name = name
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        self.lin1 = nn.Linear(500,250)
        self.lin2 = nn.Linear(250,125)
        self.lin3 = nn.Linear(125,100)
        

        self.lstm = nn.LSTM(input_size=100, hidden_size=1,
                          num_layers=num_layers, batch_first=True) #lstm


        self.h_0 = None
        self.c_0 = None
        self.hidden = None

    def forward(self, x):



        out = F.relu(self.lin1(x))
        out = F.relu(self.lin2(out))
        out = F.relu(self.lin3(out))



        if self.h_0 == None:    
            h_0 = Variable(torch.zeros(self.num_layers, 1)).to(x.device) #hidden state
            c_0 = Variable(torch.zeros(self.num_layers, 1)).to(x.device) #internal state
            self.hidden = (h_0, c_0)
        

        output, self.hidden = self.lstm(out, self.hidden) #lstm with input, hidden, and internal state

        out = torch.sigmoid(output)


        return out

# CNN -> LSTM
class Chromatin_Network6(nn.Module):
    """
    Convolutional To LSTM
    """
    def __init__(self, name, hidden_size=1, num_layers=3):
        super(Chromatin_Network6, self).__init__()
        self.name = name
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        
        self.layer_1 = nn.Conv1d(1, 3, 10) 
        self.layer_2 = nn.Conv1d(3, 5, 50) 
        self.layer_3 = nn.Conv1d(5, 10, 100)
        

        self.lstm = nn.LSTM(input_size=3430, hidden_size=1,
                          num_layers=num_layers, batch_first=True) #lstm


        self.h_0 = None
        self.c_0 = None
        self.hidden = None

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

    def forward(self, x):
        x = x.reshape(-1, 1, x.shape[1])
        x = self.relu1(self.layer_1(x))
        x = self.relu2(self.layer_2(x))
        x = self.relu3(self.layer_3(x))

        x = torch.flatten(x, start_dim=1)
        

        if self.h_0 == None:    
            h_0 = Variable(torch.zeros(self.num_layers, 1)).to(x.device) #hidden state
            c_0 = Variable(torch.zeros(self.num_layers, 1)).to(x.device) #internal state
            self.hidden = (h_0, c_0)
        

        output, self.hidden = self.lstm(x, self.hidden) #lstm with input, hidden, and internal state

        out = torch.sigmoid(output)


        return out

