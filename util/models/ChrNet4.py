import torch
from torch import nn
from util.models import ChrNet1

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
        
        # Sequential model comprising three Conv1d layers with max pooling
        self.C1D = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        # Determine input size for LSTM based on input_size
        if input_size == 500:
            lstm_in = 64 * (input_size // 8)  # Calculating output size from CNN
        elif input_size == 33000:
            lstm_in = 64 * (input_size // 8 // 8 // 8)  # Calculating output size from CNN
        else:
            raise ValueError("Invalid input_size. Supported values: 500, 33000")

        # LSTM layer taking in output of self.C1D and hidden state size
        self.lstm = nn.LSTM(input_size=lstm_in, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True) 
        
        # Fully connected layers
        self.dnn = ChrNet1.Chromatin_Network1(name, hidden_size, 1, dnn_hidden_size)

        self.hidden = None

    def forward(self, x):
        if self.hidden is None:    
            h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # Hidden state
            c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # Internal state
            self.hidden = (h_0, c_0)

        # Pass input through 1D CNN layers
        x = self.C1D(x.unsqueeze(1))  # Add channel dimension
        
        # Flatten the output of the 1D CNN
        x = x.view(x.size(0), -1)

        # Pass through the LSTM
        output, self.hidden = self.lstm(x.unsqueeze(1), self.hidden)  # Add time step dimension
        self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())

        # Pass through the DNN
        out = self.dnn(output)
        out = torch.sigmoid(out)

        return out
