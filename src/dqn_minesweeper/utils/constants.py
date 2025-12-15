from enum import Enum
from typing import Any, NamedTuple
import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Experiences(NamedTuple):
    state: Any
    action: Any
    next_state: Any
    reward: Any
    done: Any
    mask: Any
    next_mask: Any


class Identifier(Enum):
    NOTHING = 0
    BOMB = -1
    UNREVEALED = -2


NUM_TO_COLOR = {
    0: (0, 0, 0),
    1: (0, 0, 0),
    2: (0, 0, 0),
    3: (0, 0, 0),
    4: (0, 0, 0),
    5: (0, 0, 0),
    6: (0, 0, 0),
    7: (0, 0, 0),
    8: (0, 0, 0),
    -1: (255, 0, 0)
}