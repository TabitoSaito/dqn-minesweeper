import torch
import cv2
from itertools import count
from ..utils.constants import DEVICE
import numpy as np


def render_run(agent, env, run_name: str, runs: int = 10, seed=None):
    assert env.render_mode == "rgb_array"

    agent.policy_net.eval()
    for i in range(runs):
        state, info = env.reset(seed=seed)
        state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        try:
            mask = info["mask"]
            mask = torch.tensor(mask.reshape(1, -1), dtype=torch.bool, device=DEVICE)
        except KeyError:
            mask = None

        frames = []

        score = 0
        for t in count():
            action, q_values = agent.act(state, mask=mask, train_mode=False)

            frame = env.render(q_values)
            frames.append(frame)

            obs, reward, terminated, truncated, info = env.step(action.item())

            try:
                next_mask = info["mask"]
                next_mask = torch.tensor(next_mask.reshape(1, -1), dtype=torch.bool, device=DEVICE)
            except KeyError:
                next_mask = None

            score += reward
            done = terminated or truncated
            next_state = torch.tensor(
                obs, dtype=torch.float32, device=DEVICE
            ).unsqueeze(0)

            state = next_state
            mask = next_mask

            if done:
                break
        
        frame = env.render()
        frames.append(frame)

        print(f"Run {i + 1} Closed with total Reward: {score:.2f} and total Steps: {t}")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            f"src/dqn_minesweeper/replays/{run_name}_{i + 1}.mp4",
            fourcc,
            env.metadata["render_fps"],
            (frame.shape[1], frame.shape[0]),
        )
        for frame in frames:
            out.write(frame[:, :, ::-1])
        out.release()
        print(f"saved video under src/dqn_minesweeper/replays/{run_name}_{i + 1}.mp4")
    env.close()


def eval_agent(agent, env, runs=1000, print_=True):
    scores = []
    agent.policy_net.eval()
    for i in range(runs):
        state, info = env.reset(seed=i)
        state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        try:
            mask = info["mask"]
            mask = torch.tensor(mask.reshape(1, -1), dtype=torch.bool, device=DEVICE)
        except KeyError:
            mask = None

        score = 0
        for t in count():
            action, _ = agent.act(state, mask=mask, train_mode=False)
            obs, reward, terminated, truncated, info = env.step(action.item())

            try:
                next_mask = info["mask"]
                next_mask = torch.tensor(next_mask.reshape(1, -1), dtype=torch.bool, device=DEVICE)
            except KeyError:
                next_mask = None

            score += reward
            done = terminated or truncated
            next_state = torch.tensor(
                obs, dtype=torch.float32, device=DEVICE
            ).unsqueeze(0)

            state = next_state

            if done:
                break
        scores.append(score)

    avg_score = np.mean(scores)

    if print_:
        print(f"Evaluated on {runs} episodes, with Avg. Reward {avg_score:.2f}")
    return avg_score, scores