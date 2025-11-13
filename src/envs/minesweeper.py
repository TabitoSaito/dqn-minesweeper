from enum import Enum

import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

import utils.helper as helper


class Actions(Enum):
    RIGHT = 0
    RIGHT_UP = 1
    UP = 2
    UP_LEFT = 3
    LEFT = 4
    LEFT_DOWN = 5
    DOWN = 6
    DOWN_RIGHT = 7
    HIT = 8
    FLAG = 9


class Identifier(Enum):
    NOTHING = 0
    BOMB = -1
    UNREVEALED = -2
    FLAG = -3


class MinesweeperEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=8, num_bombs=10) -> None:
        self.size = size
        self.num_bombs = num_bombs
        self.window_size = 512

        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "board": spaces.Discrete(size * size),
            }
        )

        self._agent_location = np.array([-1, -1], dtype=int)
        self._board = np.zeros((size, size), dtype=int)
        self._master_board = np.zeros((size, size), dtype=int)

        self.action_space = spaces.Discrete(len(Actions), dtype=int)

        self._action_to_direction = {
            Actions.RIGHT.value: np.array([1, 0]),
            Actions.RIGHT_UP.value: np.array([1, 1]),
            Actions.UP.value: np.array([0, 1]),
            Actions.UP_LEFT.value: np.array([-1, 1]),
            Actions.LEFT.value: np.array([-1, 0]),
            Actions.LEFT_DOWN.value: np.array([-1, -1]),
            Actions.DOWN.value: np.array([0, -1]),
            Actions.DOWN_RIGHT.value: np.array([1, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"agent": self._agent_location, "board": self._board}

    def _get_info(self): ...

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # game logic
        self._board.fill(Identifier.UNREVEALED.value)
        self._master_board.fill(0)

        self._setup_master_board()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        terminated = False
        reward = -0.1 # living penalty

        match action:
            case (
                Actions.RIGHT.value
                | Actions.RIGHT_UP.value
                | Actions.UP.value
                | Actions.UP_LEFT.value
                | Actions.LEFT.value
                | Actions.LEFT_DOWN.value
                | Actions.DOWN.value
                | Actions.DOWN_RIGHT
            ):
                direction = self._action_to_direction[action]
                self._agent_location = np.clip(
                    self._agent_location + direction, 0, self.size - 1
                )
            case Actions.HIT.value:
                if self._master_board[*self._agent_location] == Identifier.BOMB.value:
                    terminated = True
                    reward -= 10
                self._reveal_cell(self._agent_location)

            case Actions.FLAG.value:
                if self._board[*self._agent_location] == Identifier.UNREVEALED.value:
                    self._board[*self._agent_location] = Identifier.FLAG.value
                if self._board[*self._agent_location] == Identifier.FLAG.value:
                    self._board[*self._agent_location] = Identifier.UNREVEALED.value

        if self._check_win():
            terminated = True
            reward += 10

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        
    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            pygame.font.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            self.font = pygame.font.SysFont('Comic Sans MS', 30)
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )

        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        for x in range(self._board.shape[0]):
            for y in range(self._board.shape[1]):
                value = self._board[x, y]
                if value != Identifier.UNREVEALED.value:
                    text_surf = self.font.render(str(value), False, (0, 0, 0))
                    text_rect = text_surf.get_rect()
                    text_rect.center = (int(pix_square_size * x + pix_square_size / 2), int(pix_square_size * y + pix_square_size / 2))
                    canvas.blit(text_surf, dest=text_rect)

        pygame.draw.rect(canvas, (255, 0, 0), pygame.Rect(pix_square_size * self._agent_location, (pix_square_size, pix_square_size)), 3)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
            np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def _setup_master_board(self):
        bomb_index = helper.generate_unique_coordinates(self.num_bombs, self.size - 1)

        self._master_board[*bomb_index] = Identifier.BOMB.value
        temp = np.pad(self._master_board, pad_width=1, mode="constant")
        for x in range(self._master_board.shape[0]):
            for y in range(self._master_board.shape[1]):
                if self._master_board[x, y] == -1:
                    continue
                min_array = temp[x : x + 3, y : y + 3]
                self._master_board[x, y] = abs(np.sum(min_array))

    def _reveal_cell(self, cell: np.ndarray):
        cells = [cell]
        while len(cells) > 0:
            reveal_all = False
            cell = cells[0]
            if self._board[*cell] != Identifier.UNREVEALED.value:
                del cells[0]
                continue
            if self._soft_reveal_cell(cell) == Identifier.NOTHING.value:
                reveal_all = True

            for i in [[1, 0], [-1, 0], [0, 1], [0, -1]]:
                if helper.index_in_bound(cell + i, self._master_board.shape):
                    master_value = self._master_board[*cell + i]
                    if master_value == Identifier.NOTHING.value or reveal_all:
                        cells.append(cell + i)
                    
            del cells[0]

    def _soft_reveal_cell(self, cell):
        value = self._master_board[*cell]
        self._board[*cell] = value
        return value
    
    def _check_win(self):
        bombs = self._board[self._master_board == Identifier.BOMB.value]
        flaged_bombs = sum(bombs == Identifier.FLAG.value)
        unrevealed = sum(self._board == Identifier.UNREVEALED.value)
        return self.num_bombs == flaged_bombs and unrevealed == 0