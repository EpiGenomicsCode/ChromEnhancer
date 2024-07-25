import torch
from torch import nn
import torch.nn.functional as F

# CNN2 -> LSTM -> DNN
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

        if self.input_size == 500:
            self.layerinput = 500
            self.kerneldepth = 5
            self.paddingrow = 2
            self.paddingcol = 3
        if self.input_size == 800:
            self.layerinput = 800
            self.kerneldepth = 8
            self.paddingrow = 3
            self.paddingcol = 3
        if self.input_size == 33000:
            self.layerinput = 66400
            self.kerneldepth = 330
            self.paddingrow = 165
            self.paddingcol = 3

        # 2D conv layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(self.kerneldepth,7), padding=(self.paddingrow,self.paddingcol))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(self.kerneldepth,7), padding=(self.paddingrow,self.paddingcol))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

        # LSTM layer that takes in self.C1D output and hidden state size
        self.lstm = nn.LSTM(input_size=self.layerinput, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True) 
        
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
        if self.input_size == 500:
            x = x.view(-1, 1, 5, 100)
        if self.input_size == 800:
            x = x.view(-1, 1, 8, 100)
        if self.input_size == 33000:
            x = x.view(-1, 1, 330, 100)

        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.shape[0], -1)

        if self.hidden == None:
            h_0 = torch.zeros(self.num_layers, self.hidden_size) #hidden state
            c_0 = torch.zeros(self.num_layers, self.hidden_size) #internal state
            self.hidden = (h_0, c_0)

        # Pass through the LSTM
        self.hidden = (self.hidden[0].to(x.device), self.hidden[1].to(x.device))
        output, self.hidden = self.lstm(x, self.hidden) #lstm with input, hidden, and internal state
        self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())
        # Pass through the DNN
        out = self.dnn(output)
        out = torch.sigmoid(out)

        return out
