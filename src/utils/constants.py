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


class Identifier(Enum):
    NOTHING = 0
    BOMB = -1
    UNREVEALED = -2
