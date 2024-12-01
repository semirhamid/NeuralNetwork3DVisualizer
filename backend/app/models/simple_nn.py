import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc0 = nn.Linear(3, 8)
        self.fc1 = nn.Linear(8, 16)  # Expanding layer
        self.fc2 = nn.Linear(16, 16)  # Core layer
        self.fc3 = nn.Linear(16, 8)  # Contracting layer
        self.fc4 = nn.Linear(8, 4)
        self.fc5 = nn.Linear(4, 2)

    def forward(self, x):
        x = torch.relu(self.fc0(x))
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x