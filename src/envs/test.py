import gymnasium as gym

def make_env():
    env = gym.make("Taxi-v3")
    return env