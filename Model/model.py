import torch
from torch import nn
import torch.nn.functional as F

#inputsize = 1
#seq_len=500
# numlayers = 2

class Chromatin_Network(nn.Module):
    
    def __init__(self, input_size=1, hidden_size=3, num_classes=1, num_layers=20):
        super(Chromatin_Network, self).__init__()
        self.num_layers = num_layers
        self.hidden_size=hidden_size
        self.mem = nn.GRU(input_size,hidden_size,num_layers,batch_first=True)
        
        self.lin1 = nn.Linear(hidden_size,num_classes)

    def forward(self, x):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # x -> (batch, seq, input)
        x = x.reshape(-1, 500, 1)

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # out -> (batch, seq, hidden)
        out, _ = self.mem(x, h0)

        out = out[:,-1,:]

        out = F.softmax(self.lin1(out))

        return torch.round(out)
