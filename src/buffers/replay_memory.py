from collections import deque
import numpy as np
import torch
from utils.constants import Experiences, DEVICE
import random


class ReplayMemory:
    def __init__(self, capacity) -> None:
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Experiences(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    

class PERMemory:
    def __init__(self, capacity, alpha=0.6, beta=0.4) -> None:
        self.memory = deque([], maxlen=capacity)
        self.priorities = deque([], maxlen=capacity)
        self.alpha = alpha
        self.beta = beta
        self.epsilon = 1e-2

    def push(self, *args, td_error=None):
        self.memory.append(Experiences(*args))
        if td_error is None:
            priority = max(self.priorities, default=1.0)
        else:
            priority = (abs(td_error) + self.epsilon) ** self.alpha
        self.priorities.append(priority)

    def sample(self, batch_size):
        priorities = np.array(self.priorities, dtype=np.float32)
        probabilities = priorities / priorities.sum()
        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        experiences = [self.memory[idx] for idx in indices]

        weights = (len(self.memory) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        return experiences, indices, torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    
    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error[0]) + self.epsilon) ** (-self.beta)
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.memory)

