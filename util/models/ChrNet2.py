import torch
from torch import nn
import torch.nn.functional as F

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

        self.inputDNN = 0
        if self.input_size == 500:
            self.inputDNN = 8000
            
        if self.input_size==4000:
            self.inputDNN = 64000

        if self.input_size == 400:
            self.inputDNN = 6400
        
        if self.input_size == 32900:
            self.inputDNN = 526336

        # Define the fully-connected layers
        self.dnn = nn.Sequential(
            nn.Linear(self.inputDNN, dnn_hidden_size),
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
