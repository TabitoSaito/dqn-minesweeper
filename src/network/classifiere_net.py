import torch.nn as nn
import torch.nn.functional as F
import torch


class ClassifierNet(nn.Module):
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


test = ClassifierNet(10, 2)
test._init_fc()
