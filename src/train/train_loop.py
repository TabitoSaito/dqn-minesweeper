from collections import deque
import numpy as np
import torch


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
        self.timesteps_on_100_episodes = deque(maxlen=100)

    def start_loop(self, agent, env, save_name = None, dyn_print = False, print_every = 100):
        print("Start Training")
        episode = 0
        try:
            while self.number_episodes <= 0 or episode < self.number_episodes:
                episode += 1
                state, _ = env.reset()
                score = 0
                for t in range(self.maximum_number_timesteps_per_episode):
                    action = agent.act(state, self.epsilon)
                    next_state, reward, done, _, _ = env.step(action)
                    agent.step(state, action, reward, next_state, done)
                    state = next_state
                    score += reward
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
