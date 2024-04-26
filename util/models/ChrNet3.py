import torch
from torch import nn
import torch.nn.functional as F
from util.models import ChrNet1

# LSTM -> DNN
class Chromatin_Network3(nn.Module):
    """
    Long Short Term Memory to Deep Neural Network
    """
    def __init__(self, name, input_size=500, hidden_size=500, num_layers=3, dnn_hidden_size=500):
        super(Chromatin_Network3, self).__init__()
        self.name = name
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.input_size = input_size
        
        # LSTM layer 
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True) 
        
        # Define the fully-connected layers
        self.dnn = ChrNet1.Chromatin_Network1(name, hidden_size, 1, dnn_hidden_size)

        self.hidden = None

        
    def forward(self, x):
        
        if self.hidden is None:    
            h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device) #hidden state
            c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device) #internal state
            self.hidden = (h_0, c_0)

        # Pass through the LSTM
        output, self.hidden = self.lstm(x, self.hidden) #lstm with input, hidden, and internal state

        # Pass through the DNN
        out = self.dnn(output)
        out = torch.sigmoid(out)

        return out
