from dqn_minesweeper.utils.helper import build_agent, save_agent, load_agent
from dqn_minesweeper.envs.minesweeper import MinesweeperEnv
from dqn_minesweeper.envs.wrappers import OneHotEncodeBoardStacked
from dqn_minesweeper.train.train_loop import prebuilt_train_loop
from dqn_minesweeper.train.evaluation import render_run
import yaml


def main():
    env = MinesweeperEnv(render_mode="rgb_array")
    env = OneHotEncodeBoardStacked(env, stack_size=1)

    state, info = env.reset()

    num_actions = env.action_space.n
    num_obs = len(state)

    with open("configs/agent/default.yaml") as stream:
        agent_config = yaml.safe_load(stream)

    agent = build_agent(agent_config, num_actions, num_obs)

    cur_agent = prebuilt_train_loop(agent, env, episodes=0)

    save_agent("test", cur_agent, agent_config, num_actions, num_obs)

    cur_agent = load_agent("test")

    render_run(cur_agent, env, run_name="test")


if __name__ == "__main__":
    main()