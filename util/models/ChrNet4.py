import torch
from torch import nn
import torch.nn.functional as F

# CNN1 -> LSTM -> DNN
class Chromatin_Network4(nn.Module):
    """
    Convolutional To LSTM To DNN
    """
    def __init__(self, name, input_size, hidden_size=500, num_layers=3, dnn_hidden_size=256):
        super(Chromatin_Network4, self).__init__()
        self.name = name
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.input_size = input_size
        
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

        if input_size == 500:
            lstmIn = 7936
        else:
            lstmIn = 64000

        # LSTM layer that takes in self.C1D output and hidden state size
        self.lstm = nn.LSTM(input_size=lstmIn, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True) 

        
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
        x = x.view(-1, 1, self.input_size)
        # Pass input through 1D CNN layers
        x = self.C1D(x)

        # Flatten the output of the 1D CNN
        x = x.view(x.size(0), -1)

        # Pass through the LSTM
        self.hidden = (self.hidden[0].to(x.device), self.hidden[1].to(x.device))
        output, self.hidden = self.lstm(x, self.hidden) #lstm with input, hidden, and internal state
        self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())
        # Pass through the DNN
        out = self.dnn(output)
        out = torch.sigmoid(out)

        return out
  