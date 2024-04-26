import torch
from torch import nn
import torch.nn.functional as F



class Chromatin_Network1(nn.Module):
    """
    A simple DNN Implementation
    :param input_size: The number of input features
    :param output_size: The number of output features
    :param dnn_hidden_size: The number of nodes in the hidden layers
    :param numLayers: The number of fully connected layers

    """
    def __init__(self, name, input_size=500, output_size=1, dnn_hidden_size=256, numLayers=3):
        super(Chromatin_Network1, self).__init__()
        self.name = name
        self.input_size = input_size
        self.output_size = output_size
        self.dnn_hidden_size = dnn_hidden_size
        self.numLayers = numLayers
        
        # Define the fully-connected layers
        layers = []
        layers.append(nn.Linear(input_size, dnn_hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.5))
        for _ in range(numLayers - 1):
            layers.append(nn.Linear(dnn_hidden_size, dnn_hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.5))
        layers.append(nn.Linear(dnn_hidden_size, output_size))
        
        self.dnn = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the model
        :param x: The input to the model
        :return: The output of the model
        """
        # Pass the output through the fully-connected layers
        x = self.dnn(x)
        # Apply sigmoid activation function to the output
        x = torch.sigmoid(x)
        
        return x