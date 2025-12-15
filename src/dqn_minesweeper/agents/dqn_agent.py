import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
from typing import Callable

from ..buffers.replay_memory import ReplayMemory, PERMemory
from ..utils.constants import DEVICE, Experiences


class BaseAgent:
    def __init__(
        self,
        action_size,
        observation_size,
        config,
        network: Callable[[int, int], nn.Module],
        noisy = False
    ) -> None:
        self.action_size = action_size
        self.config = config
        self.noisy = noisy

        self.policy_net = network(observation_size, action_size).to(DEVICE)
        self.target_net = network(observation_size, action_size).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config["LR"])
        self.criterion = nn.SmoothL1Loss()
        self.memory = ReplayMemory(config["CAPACITY"])

        self.epsilon_steps = 0
        self.update_steps = 0

    def act(self, state, mask=None, train_mode=True):
        self.policy_net.eval()
        with torch.no_grad():
            q_values = self.policy_net(state)
            if mask is not None:
                q_values[0].masked_fill_(mask[0], -1e9)
        self.policy_net.train()

        sample = random.random()

        self.epsilon = self.config["EPS_END"] + (
            self.config["EPS_START"] - self.config["EPS_END"]
        ) * math.exp(-1.0 * self.epsilon_steps / self.config["EPS_DECAY"])

        if sample > self.epsilon or not train_mode or self.noisy:
            return q_values.argmax(keepdim=True), q_values
        else:
            return torch.tensor(
                [random.sample([i for i in range(self.action_size) if q_values[0][i] > -1e9], 1)],
                device=DEVICE,
                dtype=torch.long,
            ), q_values

    def update_epsilon(self):
        self.epsilon = self.config["EPS_END"] + (
            self.config["EPS_START"] - self.config["EPS_END"]
        ) * math.exp(-1.0 * self.epsilon_steps / self.config["EPS_DECAY"])

        self.epsilon_steps += 1

    def step(self, state, action, next_state, reward, done, mask=None, next_mask=None):
        raise NotImplementedError("step method not implemented.")

    def learn(self, batch):
        raise NotImplementedError("learn method not implemented.")

    def update_net(self):
        """update target network according to tau defined in config. If tau < 1 softupdate. If tau > 1 hardupdate where tau is steps between updates."""
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()

        tau = self.config["TAU"]

        if tau < 1:
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[
                    key
                ] * tau + target_net_state_dict[key] * (1 - tau)
        elif self.update_steps >= tau:
            tau = int(tau)
            target_net_state_dict = policy_net_state_dict
            self.update_steps = 0
        else:
            self.update_steps += 0

        self.target_net.load_state_dict(target_net_state_dict)


class DQNAgent(BaseAgent):
    def __init__(
        self,
        action_size,
        observation_size,
        config,
        network: Callable[[int, int], nn.Module],
        noisy = False
    ) -> None:
        super().__init__(action_size, observation_size, config, network, noisy)

    def step(self, state, action, next_state, reward, done, mask=None, next_mask=None):
        self.memory.push(state, action, next_state, reward, done, mask, next_mask)
        if len(self.memory) > self.config["MINI_BATCH_SIZE"]:
            experiences = self.memory.sample(self.config["MINI_BATCH_SIZE"])
            batch = Experiences(*zip(*experiences))
            return self.learn(batch)

    def learn(self, batch):
        state_batch = torch.cat(batch.state)
        next_state_batch = torch.cat(batch.next_state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        done_batch = torch.cat(batch.done)

        mask_batch = torch.cat(batch.mask)
        next_mask_batch = torch.cat(batch.mask)

        try:
            self.policy_net.reset_noise()
            self.target_net.reset_noise()
        except AttributeError:
            pass

        state_action_values = self.policy_net(state_batch).masked_fill(mask_batch, -1e9).gather(1, action_batch)

        with torch.no_grad():
            next_state_action_values = self.target_net(next_state_batch).masked_fill(next_mask_batch, -1e9).max(1).values

            expected_state_action_values = (
                next_state_action_values * self.config["GAMMA"]
            ) * (1 - done_batch) + reward_batch

        loss = self.criterion(
            state_action_values, expected_state_action_values.unsqueeze(1)
        )

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()

        self.update_net()

        return loss.item()


class DQNAgentPER(BaseAgent):
    def __init__(
        self,
        action_size,
        observation_size,
        config,
        network: Callable[[int, int], nn.Module],
        noisy = False
    ) -> None:
        super().__init__(action_size, observation_size, config, network, noisy)
        self.memory = PERMemory(config["CAPACITY"], config["ALPHA"], config["BETA"])

    def step(self, state, action, next_state, reward, done):
        with torch.no_grad():
            state_action_value = self.policy_net(state)[0, action].to(DEVICE)
            next_state_action_value = self.target_net(next_state).max(1)[0].item()
            td_error = (
                reward
                + self.config["GAMMA"] * next_state_action_value * (1 - done)
                - state_action_value
            )

        self.memory.push(
            state, action, next_state, reward, done, td_error=td_error.item()
        )
        if len(self.memory) > self.config["MINI_BATCH_SIZE"]:
            experiences, indices, weights = self.memory.sample(
                self.config["MINI_BATCH_SIZE"]
            )
            batch = Experiences(*zip(*experiences))
            return self.learn(batch, indices, weights)

    def learn(self, batch, indices, weights):
        state_batch = torch.cat(batch.state)
        next_state_batch = torch.cat(batch.next_state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        done_batch = torch.cat(batch.done)

        try:
            self.policy_net.reset_noise()
            self.target_net.reset_noise()
        except AttributeError:
            pass

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        with torch.no_grad():
            next_state_action_values = self.target_net(next_state_batch).max(1).values

            expected_state_action_values = (
                next_state_action_values * self.config["GAMMA"]
            ) * (1 - done_batch) + reward_batch

        expected_state_action_values = expected_state_action_values.unsqueeze(1)

        td_errors = expected_state_action_values.detach() - state_action_values
        losses = (
            self.criterion(state_action_values, expected_state_action_values.detach())
            * weights
        )
        loss = losses.mean()

        self.memory.update_priorities(indices, td_errors.detach().cpu().numpy())

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()

        self.update_net()

        return loss.item()


class DoubleDQNAgent(BaseAgent):
    def __init__(
        self,
        action_size,
        observation_size,
        config,
        network: Callable[[int, int], nn.Module],
        noisy = False
    ) -> None:
        super().__init__(action_size, observation_size, config, network, noisy)

    def step(self, state, action, next_state, reward, done):
        self.memory.push(
            state.clone().detach(),
            action.clone().detach(),
            next_state.clone().detach(),
            reward.clone().detach(),
            done.clone().detach(),
        )
        if len(self.memory) > self.config["MINI_BATCH_SIZE"]:
            experiences = self.memory.sample(self.config["MINI_BATCH_SIZE"])
            batch = Experiences(*zip(*experiences))
            return self.learn(batch)

    def learn(self, batch):
        state_batch = torch.cat(batch.state)
        next_state_batch = torch.cat(batch.next_state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        done_batch = torch.cat(batch.done)

        try:
            self.policy_net.reset_noise()
            self.target_net.reset_noise()
        except AttributeError:
            pass

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        with torch.no_grad():
            next_actions = self.policy_net(next_state_batch).argmax(1, keepdim=True)
            next_state_action_values = (
                self.target_net(next_state_batch).gather(1, next_actions).squeeze(1)
            )

            expected_state_action_values = (
                next_state_action_values * self.config["GAMMA"]
            ) * (1 - done_batch) + reward_batch

        loss = self.criterion(
            state_action_values, expected_state_action_values.unsqueeze(1)
        )

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()

        self.update_net()
        return loss.item()


class DoubleDQNAgentPER(BaseAgent):
    def __init__(
        self,
        action_size,
        observation_size,
        config,
        network: Callable[[int, int], nn.Module],
        noisy = False
    ) -> None:
        super().__init__(action_size, observation_size, config, network, noisy)
        self.memory = PERMemory(config["CAPACITY"], config["ALPHA"], config["BETA"])

    def step(self, state, action, next_state, reward, done):
        with torch.no_grad():
            state_action_value = self.policy_net(state)[0, action].to(DEVICE)
            next_state_action_value = self.target_net(next_state).max(1)[0].item()
            td_error = reward + self.config["GAMMA"] * next_state_action_value * (1 - done) - state_action_value

        self.memory.push(state, action, next_state, reward, done, td_error=td_error.item())
        if len(self.memory) > self.config["MINI_BATCH_SIZE"]:
            experiences, indices, weights = self.memory.sample(self.config["MINI_BATCH_SIZE"])
            batch = Experiences(*zip(*experiences))
            return self.learn(batch, indices, weights)

    def learn(self, batch, indices, weights):
        state_batch = torch.cat(batch.state)
        next_state_batch = torch.cat(batch.next_state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        done_batch = torch.cat(batch.done)

        try:
            self.policy_net.reset_noise()
            self.target_net.reset_noise()
        except AttributeError:
            pass

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        with torch.no_grad():
            next_actions = self.policy_net(next_state_batch).argmax(1, keepdim=True)
            next_state_action_values = (
                self.target_net(next_state_batch).gather(1, next_actions).squeeze(1)
            )

            expected_state_action_values = (
                next_state_action_values * self.config["GAMMA"]
            ) * (1 - done_batch) + reward_batch

        expected_state_action_values = expected_state_action_values.unsqueeze(1)

        td_errors = expected_state_action_values.detach() - state_action_values
        losses = (
            self.criterion(state_action_values, expected_state_action_values.detach())
            * weights
        )
        loss = losses.mean()

        self.memory.update_priorities(indices, td_errors.detach().cpu().numpy())

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()

        self.update_net()

        return loss.item()
