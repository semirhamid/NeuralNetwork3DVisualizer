import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc0 = nn.Linear(3, 16)  # Expand more in the first hidden layer
        self.fc1 = nn.Linear(16, 32)  # Further expansion
        self.fc2 = nn.Linear(32, 32)  # Core layer for symmetry
        self.fc3 = nn.Linear(32, 16)  # Gradual contraction
        self.fc4 = nn.Linear(16, 8)   # Further contraction
        self.fc5 = nn.Linear(8, 2)    # Final contraction to output

    def forward(self, x):
        x = torch.relu(self.fc0(x))
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)  # No activation for the final layer by default
        return x
