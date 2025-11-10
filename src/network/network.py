import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self, state_size, action_size) -> None:
        super().__init__()
        self.fc1 = nn.Linear(state_size, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)