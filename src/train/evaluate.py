import cv2
import torch
from utils.constants import DEVICE
from tqdm import tqdm


def render_run(agent, env, run_name: str, max_steps: int = 0, runs: int = 10):
    assert env.render_mode in ["human", "rgb_array"]

    for i in range(runs):
        state, info = env.reset()
        mask = info["mask"]
        done = False
        score = 0
        frames = []
        j = 0
        confidence_matrix = None
        while j < max_steps or max_steps == 0:
            state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            mask = torch.tensor(mask.reshape(1, -1), dtype=torch.bool, device=DEVICE)

            j += 1
            action, confidence_matrix = agent.act(state, mask=mask[0])
            frame = env.render(confidence_matrix=confidence_matrix)
            frames.append(frame)
            state, reward, done, _, info = env.step(action.item())
            mask = info["mask"]
            score += reward

            if done:
                break

        frame = env.render()
        frames.append(frame)

        print(f"Run {i + 1} Closed with total Reward: {score:.2f}")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            f"src/replays/{run_name}_{i + 1}.mp4",
            fourcc,
            env.metadata["render_fps"],
            (frame.shape[0], frame.shape[1]),
        )
        for frame in frames:
            out.write(frame)
        out.release()
        print(f"saved video under src/replays/{run_name}_{i + 1}.mp4")
    env.close()


def eval_winrate(agent, env, max_steps: int = 0, runs: int = 50):
    wins = 0
    total_r = 0
    for i in tqdm(range(runs), desc="Evaluating", unit="Games"):
        state, info = env.reset()
        mask = info["mask"]
        done = False
        score = 0
        j = 0
        while j < max_steps or max_steps == 0:
            state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            mask = torch.tensor(mask.reshape(1, -1), dtype=torch.bool, device=DEVICE)
            j += 1
            action, _ = agent.act(state, mask=mask[0])
            state, reward, done, _, info = env.step(action.item())
            mask = info["mask"]
            score += reward

            if done:
                break

        if info["win"]:
            wins += 1
        total_r += score

    print()
    print(f"Avg. Reward: {total_r / runs:.4f}\tWinrate: {(wins / runs) * 100:.2f}%")
