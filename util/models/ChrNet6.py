import torch
from torch import nn
import torch.nn.functional as F
from util.models import ChrNet1

class Chromatin_Network6(nn.Module):
    """
    2D Convolutional To LSTM To DNN
    """
    def __init__(self, name, input_size, hidden_size=500, num_layers=3, dnn_hidden_size=256):
        super(Chromatin_Network6, self).__init__()
        self.name = name
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.input_size = input_size

        # Define the 2D convolutional layers and pooling
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Calculate the input size for LSTM based on input_size
        if input_size == 500:
            lstm_in = 800
        elif input_size == 33000:
            lstm_in = 65600
        else:
            raise ValueError("Invalid input_size. Supported values: 500, 33000")

        # LSTM layer
        self.lstm = nn.LSTM(input_size=lstm_in, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True) 
        
        # Define the fully-connected layers
        self.dnn = ChrNet1.Chromatin_Network1(name, hidden_size, 1, dnn_hidden_size)

        self.hidden = None

    def forward(self, x):
        # Reshape input based on input_size
        if self.input_size == 500:
            x = x.view(-1, 1, 100, 5)
        elif self.input_size == 33000:
            x = x.view(-1, 1, 100, 330)
        else:
            raise ValueError("Invalid input_size. Supported values: 500, 33000")

        # Pass input through convolutional layers
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the output

        # Initialize LSTM hidden state if None
        if self.hidden is None:
            self.hidden = (torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device),
                           torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device))

        # Pass through the LSTM
        output, self.hidden = self.lstm(x, self.hidden)
        self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())  # Detach hidden state

        # Pass through the DNN
        out = self.dnn(output)
        out = torch.sigmoid(out)

        return out
