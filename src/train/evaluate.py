import cv2
import torch

def render_run(agent, env, run_name: str, max_steps: int = 0, runs: int = 10):
    assert env.render_mode in ["human", "rgb_array"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for i in range(runs):
        state, info = env.reset()
        mask = info["mask"]
        done = False
        score = 0
        frames = []
        j = 0
        while (j < max_steps or max_steps == 0):
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            mask = torch.tensor(mask.reshape(1, -1), dtype=torch.bool, device=device)

            j += 1
            frame = env.render()
            frames.append(frame)
            action = agent.act(state, mask=mask[0])
            state, reward, done, _, info = env.step(action.item())
            mask = info["mask"]
            score += reward

            reward = torch.tensor([reward], dtype=torch.float32, device=device)
            action = torch.tensor([[action]], dtype=torch.long, device=device)

            if done:
                break

        frame = env.render()
        frames.append(frame)

        print(f"Run {i + 1} Closed with total Reward: {score:.2f}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(f'src/replays/{run_name}_{i + 1}.mp4', fourcc, env.metadata["render_fps"], (frame.shape[0], frame.shape[1]))
        for frame in frames:
            out.write(frame)
        print(f"saved video under src/replays/{run_name}_{i + 1}.mp4")
    env.close()
