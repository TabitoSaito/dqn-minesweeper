import torch.nn as nn
import torch.nn.functional as F
import torch

class ClassifierNet3x3(nn.Module):
    def __init__(self, in_channels, classes) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=2)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, classes)

    def _init_fc(self):
        x = torch.zeros(1, 10, 3, 3)  # (batch, channels, height, width)
        x = F.relu(self.conv1(x))

        x = x.view(1, -1).shape[1]

        print(x)

    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ClassifierNet5x5(nn.Module):
    def __init__(self, in_channels, classes) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=2)
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, classes)

    def _init_fc(self):
        x = torch.zeros(1, 10, 5, 5)  # (batch, channels, height, width)
        x = F.relu(self.conv1(x))

        x = x.view(1, -1).shape[1]

        print(x)

    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
class ClassifierNet8x8(nn.Module):
    def __init__(self, in_channels, classes) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=2)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, classes)

    def _init_fc(self):
        x = torch.zeros(1, 9, 8, 8)  # (batch, channels, height, width)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = x.view(1, -1).shape[1]

        print(x)

    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


test = ClassifierNet8x8(9, 2)
test._init_fc()
