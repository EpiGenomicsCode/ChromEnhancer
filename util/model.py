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

# CNN1 -> DNN
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
            nn.Linear(self.input_size, dnn_hidden_size),
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
        x = x.view(-1, 1, self.input_size)
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
    def __init__(self, name, input_size=500, hidden_size=500, num_layers=32,dnn_hidden_size=500):
        super(Chromatin_Network3, self).__init__()
        self.name = name
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.input_size = input_size
        
       

        # LSTM layer that takes in self.C1D output and hidden state size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True) 

        
        # Define the fully-connected layers
        self.dnn = nn.Sequential(
            nn.Linear(self.hidden_size, dnn_hidden_size),
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
            nn.Linear(dnn_hidden_size, 1)
        )


        self.h_0 = None
        self.c_0 = None
        self.hidden = None

        
    def forward(self, x):
        
        if self.h_0 == None:    
            h_0 = Variable(torch.zeros(self.num_layers, self.hidden_size)).to(x.device) #hidden state
            c_0 = Variable(torch.zeros(self.num_layers, self.hidden_size)).to(x.device) #internal state
            self.hidden = (h_0, c_0)

        
        # Reshape the input to have a channel dimension
        # Pass through the LSTM
        output, self.hidden = self.lstm(x, self.hidden) #lstm with input, hidden, and internal state
        
        # Pass through the DNN
        out = self.dnn(output)
        out = torch.sigmoid(out)

        return out
       
      

# CNN1 -> LSTM -> DNN
class Chromatin_Network4(nn.Module):
    """
    Convolutional To LSTM To DNN
    """
    def __init__(self, name, hidden_size=500, num_layers=8, dnn_hidden_size=256):
        super(Chromatin_Network4, self).__init__()
        self.name = name
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        # sequential model names C1D of three Conv1d layers with max pooling

        self.C1D = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding='same', stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding='same', stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding='same', stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        # LSTM layer that takes in self.C1D output and hidden state size
        self.lstm = nn.LSTM(input_size=7936, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True) 

        
        # Define the fully-connected layers
        self.dnn = nn.Sequential(
            nn.Linear(self.hidden_size, dnn_hidden_size),
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
            nn.Linear(dnn_hidden_size, 1)
        )


        self.h_0 = None
        self.c_0 = None
        self.hidden = None

        
    def forward(self, x):
        
        if self.h_0 == None:    
            h_0 = Variable(torch.zeros(self.num_layers, self.hidden_size)).to(x.device) #hidden state
            c_0 = Variable(torch.zeros(self.num_layers, self.hidden_size)).to(x.device) #internal state
            self.hidden = (h_0, c_0)

        
        # Reshape the input to have a channel dimension
        x = x.view(-1, 1, self.input_size)
        # Pass input through 1D CNN layers
        x = self.C1D(x)

        # Flatten the output of the 1D CNN
        x = x.view(x.size(0), -1)

        # Pass through the LSTM
        output, self.hidden = self.lstm(x, self.hidden) #lstm with input, hidden, and internal state
        
        # Pass through the DNN
        out = self.dnn(output)
        out = torch.sigmoid(out)

        return out
       
# CNN2 -> DNN
class Chromatin_Network5(nn.Module):
    def __init__(self, name):
        super(Chromatin_Network5, self).__init__()
        self.name = name
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3,3), padding=(1,1))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3,3), padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.fc1 = nn.Linear(32*25, 64)
        self.fc2 = nn.Linear(64, 1)
        
    def forward(self, x):
        x = x.view(-1, 1, 100, 5)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 32*25)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))

        return x