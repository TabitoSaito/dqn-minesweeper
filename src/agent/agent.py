import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import random
import pickle
import os
import copy

from buffer.buffer import ReplayMemory


class Agent:
    def __init__(self, action_size, config, network) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.action_size = action_size

        self.qnetwork_online = network.to(self.device)
        self.qnetwork_target = copy.deepcopy(network).to(self.device)

        self.optimizer = optim.AdamW(
            self.qnetwork_online.parameters(), lr=config["LEARNING_RATE"], amsgrad=True
        )

        self.memory = ReplayMemory(config["REPLAY_BUFFER_SIZE"], config["ALPHA"])
        self.t_step = 0
        self.config = config
        self.learn_updates = 0

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
                beta = min(1.0, self.config["BETA_START"] + self.config["BETA_INCREMENT_PER_SAMPLING"] * self.learn_updates)
                experiences = self.memory.sample(self.config["MINIBATCH_SIZE"], beta)
                self.learn(experiences)
                self.learn_updates += 1

    def act(self, state: torch.Tensor, mask: np.ndarray, epsilon=0.0):
        self.qnetwork_online.eval()
        with torch.no_grad():
            action_values = self.qnetwork_online(state)
            action_values[0].masked_fill_(mask, -1e9)
        self.qnetwork_online.train()
        if random.random() > epsilon:
            return np.argmax(action_values[0].cpu().numpy())
        else:
            valid_actions = np.where(~mask)[0]
            return random.choice(valid_actions)

    def learn(self, experiences):
        states, next_states, actions, rewards, dones, masks, next_masks, indices, weights = experiences

        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)

        curr_Q = self.qnetwork_online(states).masked_fill(masks, -1e9).gather(1, actions)

        with torch.no_grad():
            next_q_vals_online = self.qnetwork_online(next_states)
            next_q_vals_online = next_q_vals_online.masked_fill(next_masks, -1e9)
            next_actions = next_q_vals_online.argmax(dim=1, keepdim=True)

            next_q_vals_target = self.qnetwork_target(next_states)
            next_q_vals_target = next_q_vals_target.masked_fill(next_masks, -1e9)
            next_Q = next_q_vals_target.gather(1, next_actions)

            target_Q = rewards + self.config["DISCOUNT_FACTOR"] * next_Q * (1 - dones)

        td_errors = (target_Q - curr_Q).detach()

        loss_per_sample = F.smooth_l1_loss(curr_Q, target_Q.detach(), reduction='none')
        weighted_loss = (weights * loss_per_sample).mean()

        self.optimizer.zero_grad()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qnetwork_online.parameters(), 10)
        self.optimizer.step()

        with torch.no_grad():
            for param, target_param in zip(self.qnetwork_online.parameters(), self.qnetwork_target.parameters()):
                target_param.data.mul_(1 - self.config["UPDATE_RATE"])
                target_param.data.add_(self.config["UPDATE_RATE"] * param.data)

        new_priorities = td_errors.abs().squeeze(1).cpu().numpy()
        self.memory.update_priorities(indices, new_priorities)

    def save(
        self,
        epsilon,
        name=None,
    ):
        file_name = name if name else "checkpoint"
        temp = f"src/checkpoint/{file_name}.tmp"
        checkpoint = {
            "local_model_state_dict": self.qnetwork_online.state_dict(),
            "target_model_state_dict": self.qnetwork_target.state_dict(),
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

        self.qnetwork_online.load_state_dict(checkpoint["local_model_state_dict"])
        self.qnetwork_target.load_state_dict(checkpoint["target_model_state_dict"])

        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        self.memory = buffer
        return checkpoint["epsilon"]
