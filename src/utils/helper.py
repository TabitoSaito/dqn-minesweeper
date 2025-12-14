import random
import numpy as np
from agents.dqn_agent import BaseAgent, DQNAgent, DQNAgentPER, DoubleDQNAgent, DoubleDQNAgentPER
from networks.dqn_networks import DQN, DuelingDQN, NoisyDQN, NoisyDuelingDQN
import torch
import os

def get_unique_coordinates(shape: tuple[int, int], n: int):
    row, col = shape
    return [np.array([value // row, value % col], dtype=int) for value in random.sample(range(row * col), n)]

def build_agent(config, num_actions, num_obs) -> BaseAgent:
    if config["DOUBLE_DQN"] is True:
        if config["PER"] is True:
            agent_class = DoubleDQNAgentPER
        else:
            agent_class = DoubleDQNAgent
    else:
        if config["PER"] is True:
            agent_class = DQNAgentPER
        else:
            agent_class = DQNAgent

    noisy = False

    if config["DUELING"] is True and config["NOISY"] is True:
        network = NoisyDuelingDQN
        noisy = True
    elif config["DUELING"] is True:
        network = DuelingDQN
    elif config["NOISY"] is True:
        network = NoisyDQN
        noisy = True
    else:
        network = DQN

    agent = agent_class(num_actions, num_obs, config, network, noisy=noisy)

    return agent

def load_agent(name):
    content = torch.load(os.path.abspath(f"src/checkpoints/{name}.pt"), weights_only=False)
    agent = build_agent(content["config"], content["num_actions"], content["num_obs"])
    agent.policy_net.load_state_dict(content["policy_dict"])

    return agent

def save_agent(name, agent, agent_config, num_actions, num_obs):
    content = {
        "policy_dict": agent.policy_net.state_dict(),
        "config": agent_config,
        "num_actions": num_actions,
        "num_obs": num_obs,
    }

    torch.save(content, os.path.abspath(f"src/checkpoints/{name}.pt"))
    
