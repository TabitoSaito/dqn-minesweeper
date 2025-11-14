import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import pickle
import os

from replay.replay import ReplayMemory

class Agent:
    def __init__(self, action_size, config, network) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.action_size = action_size
        self.local_qnetwork = network.to(self.device)
        self.target_qnetwork = network.to(self.device)
        self.optimizer = optim.Adam(self.local_qnetwork.parameters(), lr = config["LEARNING_RATE"])
        self.memory = ReplayMemory(config["REPLAY_BUFFER_SIZE"])
        self.t_step = 0
        self.config = config

    def step(self, state, action, reward, next_state, done, mask, next_mask):
        self.memory.push((state, action, reward, next_state, done, mask, next_mask))
        self.t_step = (self.t_step + 1) % self.config["LEARN_EVERY_N_STEPS"]
        if self.t_step == 0:
            if len(self.memory.memory) > self.config["MINIBATCH_SIZE"]:
                experiences = self.memory.sample(self.config["MINIBATCH_SIZE"])
                self.learn(experiences, self.config["DISCOUNT_FACTOR"])

    def act(self, state, mask: np.ndarray | None, epsilon = 0.):
        state = np.asarray(state)
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.local_qnetwork.eval()
        with torch.no_grad():
            action_values = self.local_qnetwork(state)
            if mask is not None:
                action_values[0][mask] = -1e9
        self.local_qnetwork.train()
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            if mask is not None:
                valid_actions = np.where(~mask)[0]
            else:
                valid_actions = np.arange(self.action_size)
            return random.choice(valid_actions)
        
    def learn(self, experiences, discount_factor):
        states, next_states, actions, rewards, dones, masks, next_masks = experiences
        next_q_targets = self.target_qnetwork(next_states).masked_fill(next_masks, -1e9)
        next_q_targets = self.target_qnetwork(next_states).detach().max(1)[0].unsqueeze(1)

        q_targets = rewards + discount_factor * next_q_targets * (1 - dones)
        q_expected = self.local_qnetwork(states).gather(1, actions)
        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, epsilon, name = None, ):
        file_name = name if name else "checkpoint"
        temp = f"src/checkpoint/{file_name}.tmp"
        checkpoint = {
            'local_model_state_dict': self.local_qnetwork.state_dict(),
            'target_model_state_dict': self.target_qnetwork.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': epsilon
        }

        with open(temp, "wb") as f:
            pickle.dump((checkpoint, self.memory), f)
        os.replace(temp, f"src/checkpoint/{file_name}.pkl")
        
    def load(self, name = None):
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