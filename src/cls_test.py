from models.heat_map import BombHeatMap
from envs.minesweeper import MinesweeperEnv
from envs.wrappers import OneHotEncodeBoardStacked
from network.classifiere_net import ClassifierNet
from agent.max_select_agent import Agent
from train.evaluate import render_run, eval_winrate

env = MinesweeperEnv(size=(16, 16), num_bombs=40, render_mode="rgb_array")
env = OneHotEncodeBoardStacked(env, stack_size=1)

number_actions = env.action_space.n

net = ClassifierNet(10, 2)
heat_map = BombHeatMap(net)

# heat_map.train("5x5", 20)
# heat_map.save("5x5_test")

# heat_map.load("5x5_test")

# agent = Agent(number_actions, heat_map, kernel_size=2)

# render_run(agent, env, "test1")
# eval_winrate(agent, env)

heat_map.append_to_dataset(env, "5x5_on_16x16", repeats=20000, kernel_size=2, bombs_per_iter=20, saves_per_iter=20)
