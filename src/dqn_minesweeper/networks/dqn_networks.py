import torch.nn as nn
import torch.nn.functional as F
from networks.layers import NoisyLayer


class DQN(nn.Module):
    def __init__(self, state_size, action_size) -> None:
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DuelingDQN(nn.Module):
    def __init__(self, state_size, action_size) -> None:
        super(DuelingDQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)

        self.value_stream = nn.Linear(128, 1)
        self.advantage_stream = nn.Linear(128, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)

        return q_values


class NoisyDQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = NoisyLayer(state_size, 128)
        self.fc2 = NoisyLayer(128, 128)
        self.fc3 = NoisyLayer(128, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def reset_noise(self):
        self.fc1.reset_noise()
        self.fc2.reset_noise()
        self.fc3.reset_noise()

class NoisyDuelingDQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = NoisyLayer(state_size, 128)
        self.fc2 = NoisyLayer(128, 128)

        self.value_stream = NoisyLayer(128, 1)
        self.advantage_stream = NoisyLayer(128, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)

        return q_values

    def reset_noise(self):
        self.fc1.reset_noise()
        self.fc2.reset_noise()
        self.value_stream.reset_noise()
        self.advantage_stream.reset_noise()
