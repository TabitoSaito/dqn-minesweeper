from gymnasium import Env, ObservationWrapper
from typing import Any
import numpy as np

class MergeBoardAgent(ObservationWrapper):
    def __init__(self, env: Env):
        super().__init__(env)

    def observation(self, observation: Any) -> Any:
        agent_position = observation["agent"]
        board = observation["board"]
        agent_board = np.zeros(board.shape)
        agent_board[*agent_position] = 1

        stacked = np.stack((board, agent_board))
        return stacked