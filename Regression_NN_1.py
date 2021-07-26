import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, n_feature):
        super().__init__()
        self.fc0 = nn.Linear(n_feature, 32)
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dropout(self.relu(self.fc0(x)))
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.relu(self.fc3(x))
        return x

class Net2(nn.Module):
    def __init__(self, n_feature):
        super().__init__()
        self.fc0 = nn.Linear(n_feature, 256)
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 1)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dropout(self.relu(self.fc0(x)))
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.relu(self.fc3(x))
        return x

class Net3(nn.Module):
    def __init__(self, n_feature):
        super().__init__()
        self.fc0 = nn.Linear(n_feature, 2048)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 1)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dropout(self.relu(self.fc0(x)))
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.relu(self.fc3(x))
        return x

class Binary_Classifier(nn.Module):
    def __init__(self, n_feature):
        super().__init__()
        self.fc0 = nn.Linear(n_feature, 8)
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = torch.sigmoid

    def forward(self, x):
        x = self.relu(self.dropout(self.fc0(x)))
        x = self.relu(self.dropout(self.fc1(x)))
        x = self.relu(self.dropout(self.fc2(x)))
        x = self.sigmoid(self.fc3(x))
        return x

if __name__ == "__main__":
    model = Binary_Classifier(49)
    print (model)
