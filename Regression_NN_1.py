import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, n_feature):
        super().__init__()
        self.fc0 = nn.Linear(n_feature, 2048)
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)
        self.bn0 = nn.BatchNorm1d(2048)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dropout(self.relu(self.bn0(self.fc0(x))))
        x = self.dropout(self.relu(self.bn1(self.fc1(x))))
        x = self.dropout(self.relu(self.bn2(self.fc2(x))))
        x = self.relu(self.fc3(x))
        return x


class Binary_Classifier(nn.Module):
    def __init__(self, n_feature):
        super().__init__()
        self.fc0 = nn.Linear(n_feature, 32)
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.sigmoid = torch.sigmoid

    def forward(self, x):
        x = self.relu(self.fc0(x))
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

class Binary_Classifier2(nn.Module):
    def __init__(self, n_feature):
        super().__init__()
        self.fc0 = nn.Linear(n_feature, 64)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = torch.sigmoid

    def forward(self, x):
        x = self.relu(self.dropout(self.fc0(x)))
        x = self.relu(self.dropout(self.fc1(x)))
        x = self.relu(self.dropout(self.fc2(x)))
        x = self.sigmoid(self.fc3(x))
        return x
