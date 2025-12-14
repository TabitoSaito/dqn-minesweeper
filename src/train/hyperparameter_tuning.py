from train.train_loop import TrainLoop
from train.evaluation import eval_agent
import optuna
from functools import partial
from typing import Optional, Iterable
import numpy as np
import copy
import torch

from utils.helper import build_agent


def objective(trial, config, env, max_episodes: int = 2000, min_episodes: int = 2000, loops: int = 5, patience: int = 3, min_progress: float = 0.02, seeds: Optional[Iterable[int]] = None):
    assert max_episodes > 0, "episodes argument has to be bigger than 0"

    if seeds is None:
        seeds = [i for i in range(10)]

    converted_config = {}
    buffer_dict = {}
    for k, v in config.items():
        if str(v["condition"]) != "None":
            buffer_dict[k] = v
            continue
        match str(v["type"]).lower():
            case "float":
                converted_config[k] = trial.suggest_float(str(k), v["value"]["low"], v["value"]["high"])
            case "log_float":
                converted_config[k] = trial.suggest_float(str(k), v["value"]["low"], v["value"]["high"], log=True)
            case "int":
                converted_config[k] = trial.suggest_int(str(k), v["value"]["low"], v["value"]["high"])
            case "log_int":
                converted_config[k] = trial.suggest_int(str(k), v["value"]["low"], v["value"]["high"], log=True)
            case "list":
                converted_config[k] = trial.suggest_categorical(str(k), v["value"])

    for k, v in buffer_dict.items():
        if converted_config[v["condition"]] is True:
            match str(v["type"]).lower():
                case "float":
                    converted_config[k] = trial.suggest_float(str(k), v["value"]["low"], v["value"]["high"])
                case "log_float":
                    converted_config[k] = trial.suggest_float(str(k), v["value"]["low"], v["value"]["high"], log=True)
                case "int":
                    converted_config[k] = trial.suggest_int(str(k), v["value"]["low"], v["value"]["high"])
                case "log_int":
                    converted_config[k] = trial.suggest_int(str(k), v["value"]["low"], v["value"]["high"], log=True)
                case "list":
                    converted_config[k] = trial.suggest_categorical(str(k), v["value"])

    aucs = []
    scores = []
    train_len = []

    torch.manual_seed(0)

    state, info = env.reset()

    num_actions = env.action_space.n
    num_obs = len(state)

    for loop_idx in range(loops):
        cur_env = copy.deepcopy(env)

        cur_agent = build_agent(converted_config, num_actions, num_obs)

        loop = TrainLoop(cur_agent, cur_env, seeds=seeds, dyn_print=False, plot=False)

        prune_count = 0
        while loop.cur_episode < max_episodes:
            loop.episode_step()

            if loop.cur_episode < 200:
                continue
            if loop.cur_episode < min_episodes:
                continue
            if loop.cur_episode % 50 != 0:
                continue
            
            if (np.mean(loop.scores[-200: -100]) - np.mean(loop.scores[-100:])) / np.mean(loop.scores[-200: -100]) < min_progress:
                prune_count += 1
            else:
                prune_count = 0

            if prune_count >= patience:
                break

        eval_score, _ = eval_agent(cur_agent, cur_env, runs=200, print_=False)

        auc = np.trapezoid(loop.scores) / loop.cur_episode

        aucs.append(auc)
        scores.append(eval_score)
        train_len.append(loop.cur_episode)
        trial.set_user_attr("Episodes", train_len)

    aucs.sort()
    scores.sort()

    trial.set_user_attr("Episodes", train_len)

    return np.median(aucs), np.median(scores)

def optimize_agent(n_trials: int, config, env, max_episodes: int = 2000, min_episodes: int = 200, loops: int = 5, patience: int = 3, min_progress: float = 0.02, seeds: Optional[Iterable[int]] = None, name: Optional[str] = None):
    study = optuna.create_study(directions=["maximize", "maximize"], storage="sqlite:///instance/db.sqlite3", study_name=name)

    par_objective = partial(objective, config=config, env=env, max_episodes=max_episodes, min_episodes=min_episodes, loops=loops, patience=patience, min_progress=min_progress, seeds=seeds)

    study.optimize(par_objective, n_trials=n_trials, show_progress_bar=True)
    