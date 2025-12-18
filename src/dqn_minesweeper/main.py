from dqn_minesweeper.utils.helper import build_agent, save_agent, load_agent
from dqn_minesweeper.envs.minesweeper import MinesweeperEnv
from dqn_minesweeper.envs.wrappers import OneHotEncodeBoardStacked
from dqn_minesweeper.train.train_loop import prebuilt_train_loop
from dqn_minesweeper.train.evaluation import render_run, eval_agent
from dqn_minesweeper.train.hyperparameter_tuning import optimize_agent
import yaml


def main():
    env = MinesweeperEnv(render_mode="rgb_array", size=(8, 8), num_bombs=10)
    env = OneHotEncodeBoardStacked(env, stack_size=1)  

    state, info = env.reset()

    num_actions = env.action_space.n
    num_obs = len(state)

    # with open("configs/hyperparameter_tuning/default.yaml") as stream:
    #     config = yaml.safe_load(stream)

    # optimize_agent(100, config, env, min_progress=0.01, name="minesweeper_test", max_episodes=500)

    with open("configs/agent/default.yaml") as stream:
        agent_config = yaml.safe_load(stream)

    agent = build_agent(agent_config, num_actions, num_obs)

    cur_agent = prebuilt_train_loop(agent, env, episodes=0)

    eval_agent(cur_agent, env, runs=1000)

    save_agent("test", cur_agent, agent_config, num_actions, num_obs)

    cur_agent = load_agent("test")

    render_run(cur_agent, env, run_name="test")


if __name__ == "__main__":
    main()