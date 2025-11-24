from models.heat_map import BombHeatMap
from envs.minesweeper import MinesweeperEnv
from envs.wrappers import OneHotEncodeBoardStacked
from network.classifiere_net import ClassifierNet3x3, ClassifierNet5x5
from agent.max_select_agent import Agent
from train.evaluate import render_run, eval_winrate

env = MinesweeperEnv(size=(16, 16), num_bombs=40, render_mode="rgb_array")
env = OneHotEncodeBoardStacked(env, stack_size=1)

number_actions = env.action_space.n

net = ClassifierNet5x5(9, 2)
heat_map = BombHeatMap(net)

# heat_map.append_to_dataset(env, "5x5_on_8x8_no_nei", repeats=10000, kernel_size=2, max_bombs_per_iter=10, max_saves_per_iter=10)

# heat_map.train("5x5_on_8x8_no_nei", 20)
# heat_map.save("5x5_on_8x8_no_nei_test")

heat_map.load("5x5_on_8x8_no_nei_test")
# heat_map.evaluate("5x5_on_8x8_no_nei")

agent = Agent(number_actions, heat_map, kernel_size=2)

render_run(agent, env, "test1")
eval_winrate(agent, env, runs=1000)
