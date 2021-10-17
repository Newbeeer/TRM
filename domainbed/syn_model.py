import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

class mlp(nn.Module):
    """
    the common architecture for the left model
    """
    def __init__(self,dim):
        super(mlp, self).__init__()

        latent_dim = 128
        self.fc1 = nn.Linear(dim,latent_dim)
        self.bn1 = nn.BatchNorm1d(latent_dim)
        self.fc2 = nn.Linear(latent_dim, 2)
        self.bn2 = nn.BatchNorm1d(dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.fc2(x)

        return x

class mlp_linear(nn.Module):
    """
    the common architecture for the left model
    """
    def __init__(self,dim):
        super(mlp_linear, self).__init__()
        print("FC dim:", dim)
        self.fc = nn.Linear(dim, 1)
        #print("sanity:", self.fc.weight.data.size())
        self.fc.weight.data = torch.ones((1, dim)) / float(np.sqrt(2))
    def forward(self, x):
        #print(x.size())
        return self.fc(x)