import torch
from torch import nn
import torch.nn.functional as F

#inputsize = 1
#seq_len=500
# numlayers = 2

class Chromatin_Network(nn.Module):
    
    def __init__(self, input_size=1, hidden_size=30, num_classes=1, num_layers=30):
        super(Chromatin_Network, self).__init__()
        self.num_layers = num_layers
        self.hidden_size=hidden_size

        self.mem = nn.LSTM(input_size, hidden_size, num_layers,batch_first=True, dropout=.02)
        
        self.lin1 = nn.Linear(hidden_size,25)
        self.lin2 = nn.Linear(25,10)
        self.lin3 = nn.Linear(10,5)
        self.lin4 = nn.Linear(5,1)

    def forward(self, x):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # x -> (batch, seq, input)
        x = x.reshape(-1, 500, 1)

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # out -> (batch, seq, hidden)
        out, _ = self.mem(x, (h0,c0))
        out = out[:,-1,:]

        out = F.relu(self.lin1(out))
        out = F.relu(self.lin2(out))
        out = F.relu(self.lin3(out))
        out = F.softmax(self.lin4(out), dim=1)

        return torch.round(out)
