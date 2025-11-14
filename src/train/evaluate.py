import cv2

def render_run(agent, env, run_name: str, max_steps: int = 0):
    assert env.render_mode in ["human", "rgb_array"]
    state, info = env.reset()
    mask = info["mask"]
    done = False
    score = 0
    frames = []
    i = 0
    while not done and (i < max_steps or max_steps == 0):
        i += 1
        frame = env.render()
        frames.append(frame)
        action = agent.act(state, mask=mask)
        state, reward, done, _, info = env.step(action.item())
        mask = info["mask"]
        score += reward

    env.close()
    print(f"Closed with total Reward: {score:.2f}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f'src/train/runs_mp4/{run_name}.mp4', fourcc, env.metadata["render_fps"], (frame.shape[0], frame.shape[1]))
    for frame in frames:
        out.write(frame)
    print(f"saved video under src/train/runs_mp4/{run_name}.mp4")
    print(len(frames))
