from enum import Enum, auto
from typing import Any, NamedTuple
import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Experiences(NamedTuple):
    state: Any
    action: Any
    next_state: Any
    reward: Any
    done: Any


class Actions(Enum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3


class FruitStatus(Enum):
    """Identifier for Fruit status. Status codes have the shape 1xx"""

    @staticmethod
    def _generate_next_value_(
        name: str, start: int, count: int, last_values: list[Any]
    ) -> Any:
        return 100 + count

    RIPE = auto()
    UNRIPE = auto()
