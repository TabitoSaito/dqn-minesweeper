import yaml
from train.train_loop import TrainLoop
from agent.agent import Agent
from network.network import Network
from envs.minesweeper import MinesweeperEnv
from train.evaluate import render_run
from envs.wrappers import MergeBoardAgent

with open("configs/default.yaml") as stream:
    config = yaml.safe_load(stream)

board_size = 8
num_bombs = 10

env = MinesweeperEnv(size=board_size, num_bombs=num_bombs)
env = MergeBoardAgent(env)

board, _ = env.reset()


number_actions = env.action_space.n

network = Network(10, number_actions)

agent = Agent(number_actions, config, network)

epsilon = None
# epsilon = agent.load()

loop = TrainLoop(config, epsilon)

loop.start_loop(agent, env, dyn_print=True)

env = MinesweeperEnv(size=board_size, num_bombs=num_bombs, render_mode="rgb_array")
env = MergeBoardAgent(env)

render_run(agent, env, "test1", max_steps=config["MAXIMUM_NUMBER_TIMESTEPS_PER_EPISODE"])
