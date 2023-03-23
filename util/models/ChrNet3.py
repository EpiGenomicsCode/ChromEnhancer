import torch
from torch import nn
import torch.nn.functional as F

# LSTM -> DNN
class Chromatin_Network3(nn.Module):
    """
    Long Short Term Memory to Deep Neural Network
    """
    def __init__(self, name, input_size=500, hidden_size=500, num_layers=3,dnn_hidden_size=500):
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


        self.hidden = None

        
    def forward(self, x):
        
        if self.hidden == None:    
            h_0 = torch.zeros(self.num_layers, self.hidden_size) #hidden state
            c_0 = torch.zeros(self.num_layers, self.hidden_size) #internal state
            self.hidden = (h_0, c_0)

        
        # Reshape the input to have a channel dimension
        # Pass through the LSTM
        self.hidden = (self.hidden[0].to(x.device), self.hidden[1].to(x.device))
        output, self.hidden = self.lstm(x, self.hidden) #lstm with input, hidden, and internal state
        self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())
        # Pass through the DNN
        out = self.dnn(output)
        out = torch.sigmoid(out)

        return out
   