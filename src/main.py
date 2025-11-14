import yaml
from train.train_loop import TrainLoop
from agent.agent import Agent
from network.network import Network
from envs.minesweeper import MinesweeperEnv
from train.evaluate import render_run

with open("configs/default.yaml") as stream:
    config = yaml.safe_load(stream)

env = MinesweeperEnv(size=8, num_bombs=10)

state_size = env.observation_space.shape[0]
number_actions = env.action_space.n

network = Network(1, number_actions)

agent = Agent(number_actions, config, network)

epsilon = None
# epsilon = agent.load()

loop = TrainLoop(config, epsilon)

loop.start_loop(agent, env, dyn_print=True)

env = MinesweeperEnv(render_mode="rgb_array", size=8, num_bombs=10)

render_run(agent, env, "test1", max_steps=1000)
