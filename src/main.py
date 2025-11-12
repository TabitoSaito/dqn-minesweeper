import yaml
from train.train_loop import TrainLoop
from agent.agent import Agent
from envs.test import make_env
from network.network import Network
from envs.minesweeper import MinesweeperEnv

# with open("configs/default.yaml") as stream:
#     config = yaml.safe_load(stream)

# env = make_env()
# # state_size = env.observation_space
# number_actions = env.action_space.n

# network = Network(1, number_actions)

# agent = Agent(number_actions, config, network)

# epsilon = None
# epsilon = agent.load()

# loop = TrainLoop(config, epsilon)

# loop.start_loop(agent, env, dyn_print=True)


test = MinesweeperEnv(render_mode="human")

test.reset()
test.step(8)

# while True:
#     i = input()
#     try:
#         i = int(i)
#     except ValueError:
#         continue
#     print(test.step(i))

test._check_win()
