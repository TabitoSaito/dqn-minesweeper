from models.heat_map import BombHeatMapKernel, BombHeatMapBoard
from envs.minesweeper import MinesweeperEnv
from envs.wrappers import OneHotEncodeBoardStacked
from network.classifiere_net import ClassifierNet3x3, ClassifierNet5x5, ClassifierNet8x8
from agent.max_select_agent import Agent
from train.evaluate import render_run, eval_winrate

env = MinesweeperEnv(size=(8, 8), num_bombs=10, render_mode="rgb_array")
env = OneHotEncodeBoardStacked(env, stack_size=1)

number_actions = env.action_space.n

# ----- Kernel ----- #

# net = ClassifierNet5x5(9, 2)
# heat_map = BombHeatMapKernel(net)

# heat_map.append_to_dataset(env, "5x5_on_8x8_no_nei", repeats=10000, kernel_size=2, max_bombs_per_iter=10, max_saves_per_iter=10)

# heat_map.train("5x5_on_8x8_no_nei", 20)
# heat_map.save("5x5_on_8x8_no_nei_test")

# heat_map.load("5x5_on_8x8_no_nei_test")
# heat_map.evaluate("5x5_on_8x8_no_nei")

# ----- Board ----- #

net = ClassifierNet8x8(9, number_actions)
heat_map = BombHeatMapBoard(net)

heat_map.append_to_dataset(env, "8x8_board", repeats=50000)

heat_map.train("8x8_board", 20)
heat_map.save("8x8_board_test")

heat_map.load("8x8_board_test")
# heat_map.evaluate("8x8_board")

# ----- Evaluation ----- #

agent = Agent(number_actions, heat_map, kernel_size=2)

render_run(agent, env, "test1")
eval_winrate(agent, env, runs=1000)