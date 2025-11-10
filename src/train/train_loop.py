from collections import deque
import numpy as np
import torch

class TrainLoop:
    def __init__(self, config) -> None:
        self.number_episodes = config["NUMBER_EPISODES"]
        self.maximum_number_timesteps_per_episode = config["MAXIMUM_NUMBER_TIMESTEPS_PER_EPISODE"]
        self.epsilon_starting_value  = config["EPSILON_STARTING_VALUE"]
        self.epsilon_ending_value  = config["EPSILON_ENDING_VALUE"]
        self.epsilon_decay_value  = config["EPSILON_DECAY_VALUE"]
        self.epsilon = self.epsilon_starting_value
        self.scores_on_100_episodes = deque(maxlen = 100)

    def start_loop(self, agent, env):
        for episode in range(1, self.number_episodes + 1):
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
            self.epsilon = max(self.epsilon_ending_value, self.epsilon_decay_value * self.epsilon)
            print(f'\rEpisode {episode}\tAverage Score: {np.mean(self.scores_on_100_episodes):.2f}', end = "")
            if episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(self.scores_on_100_episodes)))
            
        torch.save(agent.local_qnetwork.state_dict(), 'checkpoint/checkpoint.pth')
        print("Finished. Agent saved under 'checkpoint/checkpoint.pth'")
        