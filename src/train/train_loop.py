from collections import deque
import numpy as np
from agent.agent import Agent
from envs.minesweeper import MinesweeperEnv
import torch

class TrainLoop:
    def __init__(self, config, epsilon=None) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.number_episodes = config["NUMBER_EPISODES"]
        self.maximum_number_timesteps_per_episode = config[
            "MAXIMUM_NUMBER_TIMESTEPS_PER_EPISODE"
        ]
        self.epsilon_starting_value = epsilon if epsilon else config["EPSILON_STARTING_VALUE"]
        self.epsilon_ending_value = config["EPSILON_ENDING_VALUE"]
        self.epsilon_decay_value = config["EPSILON_DECAY_VALUE"]
        self.epsilon = self.epsilon_starting_value
        self.scores_on_100_episodes = deque(maxlen=100)
        self.timesteps_on_100_episodes = deque(maxlen=100)

    def start_loop(self, agent: Agent, env: MinesweeperEnv, save_name = None, dyn_print = False, print_every = 100):
        print("Start Training")
        episode = 0
        try:
            while self.number_episodes <= 0 or episode < self.number_episodes:
                episode += 1
                state, info = env.reset()
                mask = info["mask"]

                state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                mask = torch.tensor(mask.reshape(1, -1), dtype=torch.bool, device=self.device)

                score = 0
                for t in range(self.maximum_number_timesteps_per_episode):
                    action = agent.act(state, mask=mask[0], epsilon = self.epsilon)
                    next_state, reward, done, _, info = env.step(action)
                    if t == 0 and done:
                        break
                    next_mask = info["mask"]
                    if t >= self.maximum_number_timesteps_per_episode - 1:
                        reward += -1

                    next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
                    reward = torch.tensor([reward], dtype=torch.float32, device=self.device)
                    done = torch.tensor([done], dtype=torch.long, device=self.device)
                    next_mask = torch.tensor(next_mask.reshape(1, -1), dtype=torch.bool, device=self.device)
                    action = torch.tensor([[action]], dtype=torch.long, device=self.device)

                    agent.step(state, action, reward, next_state, done, mask, next_mask)
                    state = next_state
                    mask = next_mask
                    score += reward[0]
                    if done:
                        break
                self.scores_on_100_episodes.append(score)
                self.timesteps_on_100_episodes.append(t)
                self.epsilon = max(
                    self.epsilon_ending_value, self.epsilon_decay_value * self.epsilon
                )
                if dyn_print:
                    print(
                        f"\rEpisode {episode}\tAverage Score: {np.mean(self.scores_on_100_episodes):.2f}\t\tAverage Timesteps: {np.mean(self.timesteps_on_100_episodes):.2f}",
                        end="",
                    )
                if episode % print_every == 0:
                    print(
                        f"\rEpisode {episode}\tAverage Score: {np.mean(self.scores_on_100_episodes):.2f}\t\tAverage Timesteps: {np.mean(self.timesteps_on_100_episodes):.2f}"
                    )
        except KeyboardInterrupt:
            pass

        if episode % print_every != 0 and dyn_print:
            print("")

        save_name = save_name if save_name else "checkpoint"
        agent.save(self.epsilon, save_name)
        print(f"\rAgent saved under 'src/checkpoint/{save_name}.tar'")
