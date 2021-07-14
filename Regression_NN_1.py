import torch
import torch.nn.functional as F
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, n_feature, n_hidden):
        super().__init__()
        self.hidden1 = nn.Linear(n_feature, n_hidden)
        self.hidden2 = nn.Linear(n_hidden, n_hidden)
        self.oupt = nn.Linear(n_hidden, 1)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.oupt(x)
        return x


model = Net(12, 64)
