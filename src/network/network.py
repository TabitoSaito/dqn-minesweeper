import torch.nn as nn
import torch.nn.functional as F
import torch

class Network(nn.Module):
    def __init__(self, in_channels, action_size) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2, stride=1)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, action_size)

    def _init_fc(self):
        x = torch.zeros(1, 1, 8, 8)   # (batch, channels, height, width)
        x = self.conv1(x)
        x = self.conv2(x)

        x = x.view(1, -1).shape[1]

        print(x)

    def forward(self, state):
        x = self.conv1(state)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)