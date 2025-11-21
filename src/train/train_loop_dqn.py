from collections import deque
import numpy as np
from agent.dqn_agent import Agent
from envs.minesweeper import MinesweeperEnv
import torch
import matplotlib.pyplot as plt
from utils.constants import DEVICE

class TrainLoop:
    def __init__(self, config, epsilon=None) -> None:
        self.number_episodes = config["NUMBER_EPISODES"]
        self.maximum_number_timesteps_per_episode = config[
            "MAXIMUM_NUMBER_TIMESTEPS_PER_EPISODE"
        ]
        self.epsilon_starting_value = epsilon if epsilon else config["EPSILON_STARTING_VALUE"]
        self.epsilon_ending_value = config["EPSILON_ENDING_VALUE"]
        self.epsilon_decay_value = config["EPSILON_DECAY_VALUE"]
        self.epsilon = self.epsilon_starting_value
        self.scores_on_100_episodes = deque(maxlen=100)
        self.wins_on_100_episodes = deque(maxlen=100)

        self.mean_score_on_100_episodes = []

    def start_loop(self, agent: Agent, env: MinesweeperEnv, save_name = None, dyn_print = False, print_every = 100):
        print("Start Training")
        episode = 0

        plt.ion()

        x = [(i + 1) * 100 for i in range(episode // 100)]
        y = self.mean_score_on_100_episodes

        color1 = "b"
        color2 = "r"

        graph1 = plt.plot(x, y, color=color1)[0]
        graph2 = plt.plot(x, y, color=color2)[0]

        plt.title("training...")
        plt.xlabel("episodes")
        plt.ylabel("average reward")

        try:
            while self.number_episodes <= 0 or episode < self.number_episodes:
                episode += 1
                state, info = env.reset()
                mask = info["mask"]

                state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                mask = torch.tensor(mask.reshape(1, -1), dtype=torch.bool, device=DEVICE)

                score = 0
                for t in range(self.maximum_number_timesteps_per_episode):
                    action = agent.act(state, mask=mask[0], epsilon = self.epsilon)
                    next_state, reward, done, _, info = env.step(action)
                    next_mask = info["mask"]
                    score += reward

                    next_state = torch.tensor(next_state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                    reward = torch.tensor([reward], dtype=torch.float32, device=DEVICE)
                    done = torch.tensor([done], dtype=torch.long, device=DEVICE)
                    next_mask = torch.tensor(next_mask.reshape(1, -1), dtype=torch.bool, device=DEVICE)
                    action = torch.tensor([[action]], dtype=torch.long, device=DEVICE)

                    agent.step(state, action, reward, next_state, done, mask, next_mask)
                    state = next_state
                    mask = next_mask
                    if done:
                        break

                agent.reset()

                self.scores_on_100_episodes.append(score)
                self.wins_on_100_episodes.append(int(info["win"]))
                self.epsilon = max(
                    self.epsilon_ending_value, self.epsilon_decay_value * self.epsilon
                )
                if dyn_print:
                    print(
                        f"\rEpisode {episode}\t\tAverage Score: {np.mean(self.scores_on_100_episodes):.2f}\t\tWin Rate: {np.mean(self.wins_on_100_episodes) * 100:.2f}%",
                        end="",
                    )
                if episode % print_every == 0:
                    print(
                        f"\rEpisode {episode}\t\tAverage Score: {np.mean(self.scores_on_100_episodes):.2f}\t\tWin Rate: {np.mean(self.wins_on_100_episodes) * 100:.2f}%"
                    )
                    self.mean_score_on_100_episodes.append(np.mean(self.scores_on_100_episodes))

                    x = [(i + 1) * 100 for i in range(episode // 100)]
                    y = self.mean_score_on_100_episodes

                    z = np.polyfit(x, y, 1)
                    p = np.poly1d(z)

                    graph1.remove()
                    graph2.remove()

                    graph1 = plt.plot(x, y, color=color1)[0]
                    graph2 = plt.plot(x, p(x), color=color2)[0]

                    plt.pause(0.01)

        except KeyboardInterrupt:
            pass

        if episode % print_every != 0 and dyn_print:
            print("")

        save_name = save_name if save_name else "checkpoint"
        agent.save(self.epsilon, save_name)
        print(f"\rAgent saved under 'src/checkpoint/{save_name}.pkl'")

        plt.show()
