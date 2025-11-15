import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import random
import pickle
import os

from buffer.buffer import ReplayMemory


class Agent:
    def __init__(self, action_size, config, network) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.action_size = action_size
        self.local_qnetwork = network.to(self.device)
        self.target_qnetwork = network.to(self.device)
        self.optimizer = optim.AdamW(
            self.local_qnetwork.parameters(), lr=config["LEARNING_RATE"], amsgrad=True
        )
        self.memory = ReplayMemory(config["REPLAY_BUFFER_SIZE"])
        self.t_step = 0
        self.config = config

    def step(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
        mask: torch.Tensor,
        next_mask: torch.Tensor,
    ):
        self.memory.push(state, action, reward, next_state, done, mask, next_mask)
        self.t_step = (self.t_step + 1) % self.config["LEARN_EVERY_N_STEPS"]
        if self.t_step == 0:
            if len(self.memory) > self.config["MINIBATCH_SIZE"]:
                experiences = self.memory.sample(self.config["MINIBATCH_SIZE"])
                self.learn(experiences, self.config["DISCOUNT_FACTOR"])

    def act(self, state: torch.Tensor, mask: np.ndarray, epsilon=0.0):
        self.local_qnetwork.eval()
        with torch.no_grad():
            action_values = self.local_qnetwork(state)
            action_values[0][mask] = -1e9
        self.local_qnetwork.train()
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            valid_actions = np.where(~mask)[0]
            return random.choice(valid_actions)

    def learn(self, experiences, discount_factor):
        states, next_states, actions, rewards, dones, masks, next_masks = experiences
        next_q_targets = self.target_qnetwork(next_states).masked_fill(next_masks, -1e9)
        next_q_targets = (
            self.target_qnetwork(next_states).detach().max(1)[0].unsqueeze(1)
        )

        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)

        q_targets = rewards + discount_factor * next_q_targets * (1 - dones)
        q_expected = self.local_qnetwork(states).gather(1, actions)
        criterion = nn.SmoothL1Loss()
        loss = criterion(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.local_qnetwork.parameters(), 10)
        self.optimizer.step()

        local_dict = self.local_qnetwork.state_dict()
        target_dict = self.target_qnetwork.state_dict()
        for key in local_dict:
            target_dict[key] = local_dict[key] * self.config[
                "UPDATE_RATE"
            ] + target_dict[key] * (1 - self.config["UPDATE_RATE"])
        self.target_qnetwork.load_state_dict(target_dict)

    def save(
        self,
        epsilon,
        name=None,
    ):
        file_name = name if name else "checkpoint"
        temp = f"src/checkpoint/{file_name}.tmp"
        checkpoint = {
            "local_model_state_dict": self.local_qnetwork.state_dict(),
            "target_model_state_dict": self.target_qnetwork.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": epsilon,
        }

        with open(temp, "wb") as f:
            pickle.dump((checkpoint, self.memory), f)
        os.replace(temp, f"src/checkpoint/{file_name}.pkl")

    def load(self, name=None):
        file_name = name if name else "checkpoint"
        path = f"src/checkpoint/{file_name}.pkl"
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path, "rb") as f:
            checkpoint, buffer = pickle.load(f)

        self.local_qnetwork.load_state_dict(checkpoint["local_model_state_dict"])
        self.target_qnetwork.load_state_dict(checkpoint["target_model_state_dict"])

        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        self.memory = buffer
        return checkpoint["epsilon"]
