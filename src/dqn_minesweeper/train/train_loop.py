import torch
from itertools import count, cycle
import numpy as np
import copy
from multiprocessing import Process, Queue
from typing import Iterable, Optional
import time
from tqdm import tqdm
from ..utils.constants import DEVICE
from .plots import plot_training
from ..utils.helper import build_agent, save_agent
from .evaluation import eval_agent


class TrainLoop:
    def __init__(self, agent, env, seeds: Optional[Iterable[int]] = None, dyn_print=True, plot=True) -> None:
        self.agent = agent
        self.env = env

        self.dyn_print = dyn_print
        self.plot = plot

        self.scores = []
        self.steps = []
        self.best_avg_reward = -float("inf")

        self.queue1 = Queue(maxsize=1000)
        self.queue2 = Queue(maxsize=1000)

        self.seeds = cycle(seeds) if seeds is Iterable else None

        if plot:
            self.p = Process(target=plot_training, args=(self.queue1, self.queue2,), daemon=False)
            self.p.start()

        self.cur_episode = 0
        self.best_episode = 0

    def episode_step(self):
        self.cur_episode += 1

        seed = next(self.seeds) if self.seeds is Iterable else None

        state, info = self.env.reset(seed=seed)
        try:
            mask = info["mask"]
            mask = torch.tensor(mask.reshape(1, -1), dtype=torch.bool, device=DEVICE)
        except KeyError:
            mask = None

        state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        self.agent.update_epsilon()

        score = 0
        q_values_buffer = []
        q_values_mean = []

        start_time = time.time()
        for t in count():
            action, q_values = self.agent.act(state, mask=mask)
            obs, reward, terminated, truncated, info = self.env.step(action.item())
            try:
                next_mask = info["mask"]
                next_mask = torch.tensor(next_mask.reshape(1, -1), dtype=torch.bool, device=DEVICE)
            except KeyError:
                next_mask = None

            q_values_buffer.append(q_values)

            score += reward
            reward = torch.tensor([reward], device=DEVICE)
            done = terminated or truncated
            done = torch.tensor([done], device=DEVICE, dtype=torch.float32)

            next_state = torch.tensor(
                obs, dtype=torch.float32, device=DEVICE
            ).unsqueeze(0)

            loss = self.agent.step(state, action, next_state, reward, done)

            state = next_state
            mask = next_mask

            if self.plot:
                self.queue2.put((loss))

            if terminated or truncated:
                break

        step_duration = t / (time.time() - start_time)

        self.scores.append(score)
        self.steps.append(t)
        q_values_buffer = torch.cat(q_values_buffer)
        q_values_mean = q_values_buffer.mean(dim=0).tolist()

        if np.mean(self.scores[-100:]) > self.best_avg_reward and self.cur_episode > 100:
            self.best_avg_reward = np.mean(self.scores[-100:])
            self.best_episode = self.cur_episode

        if self.dyn_print:
            print(
                f"\rEpisode {self.cur_episode}\t\tAverage Score: {np.mean(self.scores[-100:]):.2f}\t\tAverage steps: {np.mean(self.steps[-100:]):.2f}\t\tEpsilon: {self.agent.epsilon:.4f}",
                end="",
            )

            if self.cur_episode % 100 == 0:
                print(
                    f"\rEpisode {self.cur_episode}\t\tAverage Score: {np.mean(self.scores[-100:]):.2f}\t\tAverage steps: {np.mean(self.steps[-100:]):.2f}\t\tEpsilon: {self.agent.epsilon:.4f}",
                )
        
        if self.plot:
            self.queue1.put((self.scores, self.best_avg_reward, q_values_mean, self.agent.epsilon, step_duration))

    def end_training(self):
        if self.cur_episode % 100 != 0 and self.dyn_print:
            print("")


def prebuilt_train_loop(agent, env, episodes=0, seeds: Optional[Iterable[int]] = None, dyn_print=True, plot=True):
    loop = TrainLoop(agent=agent, env=env, seeds=seeds, dyn_print=dyn_print, plot=plot)
    try:
        for _ in count(start=1):
            loop.episode_step()
            if episodes == 0:
                continue
            if loop.cur_episode >= episodes:
                break
    except KeyboardInterrupt:
        pass

    if loop.cur_episode % 100 != 0 and dyn_print:
        print("")


def train_best_agent(agent_config, env, name, loops=10, max_episodes=0, min_episodes=200, patience=3, min_progress=0.02):
    policy_state_dicts = []

    state, info = env.reset()

    num_actions = env.action_space.n
    num_obs = len(state)

    for loop_idx in range(loops):
        cur_env = copy.deepcopy(env)

        cur_agent = build_agent(agent_config, num_actions, num_obs)

        loop = TrainLoop(cur_agent, cur_env, dyn_print=False, plot=False)

        prune_count = 0
        while loop.cur_episode <= max_episodes or max_episodes == 0:
            print(f"\rloop: {loop_idx + 1}/{loops}\t\tcur episode: {loop.cur_episode}/{max_episodes}", end="")
            loop.episode_step()

            if loop.cur_episode % 50 == 0:
                policy_state_dicts.append(copy.deepcopy(loop.agent.policy_net.state_dict()))

            if loop.cur_episode < 200:
                continue
            if loop.cur_episode < min_episodes:
                continue
            if loop.cur_episode % 50 != 0:
                continue

            if (np.mean(loop.scores[-200: -100]) - np.mean(loop.scores[-100:])) / np.mean(loop.scores[-200: -100]) < min_progress:
                prune_count += 1
            else:
                prune_count = 0

            if prune_count >= patience:
                break
        print()

    eval_state_dict = []
    for state_dict in tqdm(policy_state_dicts, desc="evaluating agents"):
        cur_agent = build_agent(agent_config, num_actions, num_obs)
        cur_agent.policy_net.load_state_dict(state_dict)

        cur_env = copy.deepcopy(env)

        score, _ = eval_agent(cur_agent, cur_env, runs=100, print_=False)
        eval_state_dict.append((score, state_dict))

    eval_state_dict.sort(key=lambda x: x[0])
    eval_state_dict = eval_state_dict[-10:]

    best = None
    best_score = -1e9
    for _, state_dict in tqdm(eval_state_dict, desc="evaluating top 10"):
        cur_agent = build_agent(agent_config, num_actions, num_obs)
        cur_agent.policy_net.load_state_dict(state_dict)

        cur_env = copy.deepcopy(env)

        score, _ = eval_agent(cur_agent, cur_env, runs=1000, print_=False)

        if score > best_score:
            best = state_dict
            best_score = score

    cur_agent = build_agent(agent_config, num_actions, num_obs)
    cur_agent.policy_net.load_state_dict(best)

    save_agent(name, cur_agent, agent_config, num_actions, num_obs)

    print(f'saved agent with name "{name}" and score {best_score:.2f}')
