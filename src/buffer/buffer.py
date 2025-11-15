import torch
from typing import NamedTuple, Any
import numpy as np

class Experiences(NamedTuple):
    state: Any
    action: Any
    reward: Any
    next_state: Any
    done: Any
    mask: Any
    next_mask: Any

class ReplayMemory:
    def __init__(self, capacity, alpha) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.capacity = capacity
        self.memory = []
        self.alpha = alpha
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(Experiences(*args))
        else:
            self.memory[self.position] = Experiences(*args)

        max_prio = self.priorities.max() if len(self.memory) > 1 else 1.0
        self.priorities[self.position] = max_prio
        self.position = (self.position + 1) % self.capacity


    def sample(self, batch_size, beta = 0.4):
        prios = self.priorities[:len(self.memory)]
        probs = prios ** self.alpha
        probs = probs / sum(probs)

        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]

        batch = Experiences(*zip(*samples))

        total = len(self.memory)
        weights = (total * probs[indices]) ** (-beta)
        weights = weights / weights.max()
        weights = torch.as_tensor(weights, dtype=torch.float32).unsqueeze(1).to(self.device)

        state_batch = torch.cat(batch.state)
        next_state_batch = torch.cat(batch.next_state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        done_batch = torch.cat(batch.done)
        mask_batch = torch.cat(batch.mask)
        next_mask_batch = torch.cat(batch.next_mask)

        return state_batch, next_state_batch, action_batch, reward_batch, done_batch, mask_batch, next_mask_batch, indices, weights
    
    def update_priorities(self, indices, priorities):
        eps = 1e-6
        for idx, pr in zip(indices, priorities):
            self.priorities[idx] = max(pr, eps)
    
    def __len__(self):
        return len(self.memory)