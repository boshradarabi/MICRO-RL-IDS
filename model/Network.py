import torch
import torch.nn as nn
import torch.nn.functional as F


# Create a network with 3 hidden layers

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, hidden=100):
        super(QNetwork,self).__init__() 
        self.seed = torch.manual_seed(seed)
        self.fc1  = nn.Linear(state_size,hidden)
        self.fc2  = nn.Linear(hidden,hidden)
        self.fc3  = nn.Linear(hidden,hidden)
        self.fc4  = nn.Linear(hidden,action_size)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return self.fc4(x)