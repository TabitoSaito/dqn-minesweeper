import random
from ..agents.dqn_agent import BaseAgent, DQNAgent, DQNAgentPER, DoubleDQNAgent, DoubleDQNAgentPER
from ..networks.dqn_networks import DQN, DuelingDQN, NoisyDQN, NoisyDuelingDQN, DQNCNN
import torch
import os


def generate_unique_coordinates(
    n, upper_bound_x, upper_bound_y, lower_bound_x=0, lower_bound_y=0, except_=[]
):
    cords = []

    for _ in range(n):
        while True:
            x = random.randint(lower_bound_x, upper_bound_x)
            y = random.randint(lower_bound_y, upper_bound_y)

            cord = [x, y]
            if cord in cords:
                continue
            elif cord[0] == except_[0] and cord[1] == except_[1]:
                continue
            else:
                cords.append(cord)
                break
    return list(map(list, zip(*cords)))


def index_in_bound(index: tuple[int, int], bound: tuple[int, int]):
    if not 0 <= index[0] < bound[0]:
        return False
    if not 0 <= index[1] < bound[1]:
        return False
    return True


def action_to_index(action, shape: tuple):
    row = int(action / shape[1])
    col = action % shape[1]
    return (row, col)


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
    elif config["CNN"] is True:
        network = DQNCNN
    else:
        network = DQN

    agent = agent_class(num_actions, num_obs, config, network, noisy=noisy)

    return agent

def load_agent(name):
    content = torch.load(os.path.abspath(f"src/dqn_minesweeper/checkpoints/{name}.pt"), weights_only=False)
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

    torch.save(content, os.path.abspath(f"src/dqn_minesweeper/checkpoints/{name}.pt"))
    
