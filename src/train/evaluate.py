import cv2
import torch

def render_run(agent, env, run_name: str, max_steps: int = 0):
    assert env.render_mode in ["human", "rgb_array"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    state, info = env.reset()
    mask = info["mask"]
    done = False
    score = 0
    frames = []
    i = 0
    while not done and (i < max_steps or max_steps == 0):
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        mask = torch.tensor(mask.reshape(1, -1), dtype=torch.bool, device=device)

        i += 1
        frame = env.render()
        frames.append(frame)
        action = agent.act(state, mask=mask[0])
        state, reward, done, _, info = env.step(action.item())
        mask = info["mask"]
        score += reward

        reward = torch.tensor([reward], dtype=torch.float32, device=device)
        action = torch.tensor([[action]], dtype=torch.long, device=device)

    env.close()
    print(f"Closed with total Reward: {score:.2f}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f'src/train/runs_mp4/{run_name}.mp4', fourcc, env.metadata["render_fps"], (frame.shape[0], frame.shape[1]))
    for frame in frames:
        out.write(frame)
    print(f"saved video under src/train/runs_mp4/{run_name}.mp4")
    print(len(frames))
