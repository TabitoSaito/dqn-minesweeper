import torch
from enum import Enum

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Identifier(Enum):
    NOTHING = 0
    BOMB = -1
    UNREVEALED = -2