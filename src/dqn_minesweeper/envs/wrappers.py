from gymnasium import Env, ObservationWrapper
from typing import Any
import numpy as np
from collections import deque

class OneHotEncodeBoardStacked(ObservationWrapper):
    def __init__(self, env: Env, stack_size = 4):
        super().__init__(env)
        self.channels = 9
        self.stack = deque(maxlen=stack_size * self.channels)
        self.stack_size = stack_size 

    def observation(self, observation: Any) -> Any:
        board = observation["board"]

        temp_arrays = []

        for i in range(9):
            temp = np.zeros(board.shape)
            temp[board == i] = 1
            temp_arrays.append(temp)

        if len(self.stack) < self.stack_size * self.channels:
            for i in range(self.stack_size):
                for a in temp_arrays:
                    self.stack.append(a)

        for a in temp_arrays:
            self.stack.append(a)
        
        stacked = np.stack(self.stack)
        return stacked
    
    def render(self, q_values=None):
        return self.env.render(q_values=q_values)