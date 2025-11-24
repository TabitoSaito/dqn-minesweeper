import torch
from models.heat_map import BombHeatMapKernel, BombHeatMapBoard

class Agent:
    def __init__(self, number_actions, heat_map: BombHeatMapKernel | BombHeatMapBoard, kernel_size=1) -> None:
        self.heat_map = heat_map
        self.n_actions = number_actions
        self.kernel_size = kernel_size

    def act(self, state: torch.Tensor, mask):
        prediction = self.heat_map.predict(state, kernel_size=self.kernel_size)
        save_prob = prediction[0].flatten()
        masked_prob = save_prob.masked_fill(mask, 0)
        action = torch.max(masked_prob, 0)[1]
        return action, prediction
