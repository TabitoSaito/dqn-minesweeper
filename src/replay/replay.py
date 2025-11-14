import torch
from collections import deque
import random
from collections import namedtuple
from typing import NamedTuple, Any

# Experiences = namedtuple("Experience", ("state", "action", "reward", "next_state", "done", "mask", "next_mask"))

class Experiences(NamedTuple):
    state: Any
    action: Any
    reward: Any
    next_state: Any
    done: Any
    mask: Any
    next_mask: Any

class ReplayMemory:
    def __init__(self, capacity) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Experiences(*args))

    def sample(self, batch_size):
        experiences = random.sample(self.memory, batch_size)
        batch = Experiences(*zip(*experiences))

        state_batch = torch.cat(batch.state)
        next_state_batch = torch.cat(batch.next_state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        done_batch = torch.cat(batch.done)
        mask_batch = torch.cat(batch.mask)
        next_mask_batch = torch.cat(batch.next_mask)

        return state_batch, next_state_batch, action_batch, reward_batch, done_batch, mask_batch, next_mask_batch
    
    def __len__(self):
        return len(self.memory)