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
    def __init__(self, name, input_size=500, output_size=1, dnn_hidden_size=256):
        super(Chromatin_Network1, self).__init__()
        self.name = name
        self.input_size = input_size
        self.output_size = output_size
        self.dnn_hidden_size = dnn_hidden_size

        # Define the fully-connected layers
        self.dnn = nn.Sequential(
            nn.Linear(input_size, dnn_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(dnn_hidden_size, dnn_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(dnn_hidden_size, dnn_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(dnn_hidden_size, dnn_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(dnn_hidden_size, output_size)
        )

    def forward(self, x):
        # Pass the output through the fully-connected layers
        x = self.dnn(x)
        # Apply sigmoid activation function to the output
        x = torch.sigmoid(x)
        
        return x

# CNN -> DNN
class Chromatin_Network2(nn.Module):
    """
    Convolutional Network to Deep Neural Network
    """
    def __init__(self, name, input_size=500, output_size=1, kernel_size=3, cnn_hidden_size=32, dnn_hidden_size=256):
        super(Chromatin_Network2, self).__init__()
        self.name = name
        self.input_size = input_size
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.cnn_hidden_size = cnn_hidden_size
        self.dnn_hidden_size = dnn_hidden_size

        # Define the 1D CNN layers
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=cnn_hidden_size, kernel_size=kernel_size, padding='same', stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        # Define the fully-connected layers
        self.dnn = nn.Sequential(
            nn.Linear(8000, dnn_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(dnn_hidden_size, dnn_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(dnn_hidden_size, dnn_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(dnn_hidden_size, dnn_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(dnn_hidden_size, output_size)
        )

    def forward(self, x):
        # Reshape the input to have a channel dimension
        x = x.view(-1, 1, 500)
        # Pass input through 1D CNN layers
        x = self.cnn(x)
        # Flatten the output of the 1D CNN
        x = x.view(x.size(0), -1)
        # Pass the output through the fully-connected layers
        x = self.dnn(x)
        # Apply sigmoid activation function to the output
        x = torch.sigmoid(x)
        return x

# LSTM -> DNN
class Chromatin_Network3(nn.Module):
    """
    Long Short Term Memory to Deep Neural Network
    """
    def __init__(self, name, hidden_size=64, num_layers=32):
        super(Chromatin_Network3, self).__init__()
        self.name = name
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size=500, hidden_size=hidden_size, num_layers=num_layers, batch_first=True) #lstm

        # Define the fully-connected layers
        self.dnn = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 1)
        )

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
        out = self.dnn(out)
        out = torch.sigmoid(out)

        return out

# CNN -> LSTM -> DNN
class Chromatin_Network4(nn.Module):
    """
    Convolutional To LSTM To DNN
    """
    def __init__(self, name, hidden_size=1, num_layers=32):
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

        self.dropout = nn.Dropout(0.25)

        
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
        x = self.dropout(x)
        out = torch.sigmoid(self.lin5(out))

        return out

# Transformer -> DNN
class Chromatin_Network5(nn.Module):
    def __init__(self, name, input_size=500, hidden_size=256, num_heads=10, output_size=1):
        super().__init__()
        self.name = name
        # Use nn.Transformer to compute a weighted sum of the input elements
        self.transformer = nn.Transformer(input_size, num_heads)
        self.layer_1 = nn.Linear(input_size, 500) 
        self.layer_2 = nn.Linear(500, 500) 
        self.layer_3 = nn.Linear(500, 500) 
        self.layer_4 = nn.Linear(500, 500)
        self.layer_out = nn.Linear(500, 1) 

    def forward(self, x):
        # Reshape the input to (batch_size, sequence_length, input_size)
        x = x.view(-1, 1, 500)
        # Apply the transformer to the input sequence, is this 1 to 1?
        x = self.transformer(x,x)
        # Flatten the output to (batch_size, input_size)
        x = x.view(-1, 500)
        # Apply the DNN to the transformed input
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = F.relu(self.layer_3(x))
        x = F.relu(self.layer_4(x))
        x = F.relu(self.layer_out(x))
        
        return x
