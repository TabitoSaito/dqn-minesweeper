import numpy as np
import pygame
import torch

import gymnasium as gym
from gymnasium import spaces

from ..utils import helper
from ..utils.constants import Identifier, NUM_TO_COLOR


class MinesweeperEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}

    def __init__(self, render_mode=None, size=(8, 8), num_bombs=10) -> None:
        self.size = size
        self.pix_square_size = 50
        self.num_bombs = num_bombs
        self.bomb_index = []
        self.w = size[0] * self.pix_square_size
        self.h = size[1] * self.pix_square_size
        self.window_size = max(self.w, self.h)

        self.observation_space = spaces.Dict(
            {
                "board": spaces.Discrete(size[0] * size[1]),
            }
        )

        self._board = np.zeros((size[0], size[1]), dtype=int)
        self._master_board = np.zeros((size[0], size[1]), dtype=int)

        self.action_space = spaces.Discrete(self._board.size, dtype=int)

        self._mask = np.zeros(self.action_space.n, dtype=bool)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None
        self.font = None

    def _get_obs(self):
        return {"board": self._board}

    def _get_info(self):
        return {
            "mask": self._mask,
            "master_board": self._master_board,
            "board": self._board,
            "save_actions": self._get_save_actions(),
            "bomb_actions": self._get_bomb_actions(),
            "win": self._check_win(),
        }

    def _get_save_actions(self):
        return np.where(
            (self._board.flatten() == Identifier.UNREVEALED.value)
            & (self._master_board.flatten() != Identifier.BOMB.value)
        )

    def _get_bomb_actions(self):
        return np.where(
            (self._board.flatten() == Identifier.UNREVEALED.value)
            & (self._master_board.flatten() == Identifier.BOMB.value)
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # game logic
        self._board.fill(Identifier.UNREVEALED.value)
        self._master_board.fill(0)

        self._update_mask()
        self.win = False

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):  # if action == -1, performs a random action that is save
        self._terminated = False
        self.reward = 0

        if action == -1:
            action = np.random.choice(self._get_save_actions()[0])

        row, col = helper.action_to_index(action, self._board.shape)

        if np.all(self._master_board == 0):
            self._setup_master_board([row, col])

        if self._board[row, col] != Identifier.UNREVEALED.value:
            self.reward += -0.1
        else:
            self._reveal_cell(np.array([row, col]))
            self.reward += 0.05

        if self._check_win():
            self._terminated = True
            self.reward += 1

        self._update_mask()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, self.reward, self._terminated, False, info

    def render(self, q_values=None):
        if self.render_mode == "rgb_array":
            return self._render_frame(q_values=q_values)

    def _render_frame(self, q_values=None):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        if self.font is None and self.render_mode in self.metadata["render_modes"]:
            pygame.font.init()
            self.font = pygame.font.SysFont("Comic Sans MS", 30)

        background = pygame.Surface((self.window_size, self.window_size))
        background.fill((255, 255, 255))
        canvas = pygame.Surface((self.w, self.h))
        canvas.fill((255, 255, 255))

        for x in range(self.size[1] + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, self.pix_square_size * x),
                (self.window_size, self.pix_square_size * x),
                width=3,
            )
        for y in range(self.size[0] + 1):
            pygame.draw.line(
                canvas,
                0,
                (self.pix_square_size * y, 0),
                (self.pix_square_size * y, self.window_size),
                width=3,
            )

        if q_values is not None:
            prediction = torch.softmax(q_values, dim=1).squeeze(0)
            confidence_matrix = torch.reshape(prediction, self._board.shape)
        else:
            confidence_matrix = None

        for x in range(self._board.shape[0]):
            for y in range(self._board.shape[1]):
                value = self._board[x, y]
                if value != Identifier.UNREVEALED.value:
                    text_surf = self.font.render(str(value), False, NUM_TO_COLOR[int(value)])
                    text_rect = text_surf.get_rect()
                    text_rect.center = (
                        int(self.pix_square_size * x + self.pix_square_size / 2),
                        int(self.pix_square_size * y + self.pix_square_size / 2),
                    )
                    canvas.blit(text_surf, dest=text_rect)
                else:
                    if confidence_matrix is not None:
                        
                        confidence_cell = confidence_matrix[x, y]

                        confidence_amount = (confidence_cell - torch.min(confidence_matrix[confidence_matrix != 0])) / (torch.max(confidence_matrix[confidence_matrix != 0]) - torch.min(confidence_matrix[confidence_matrix != 0]))

                        if confidence_amount == float("-inf"):
                            continue

                        if torch.isnan(confidence_amount) > 0:
                            confidence_amount = 0.5

                        if confidence_cell == torch.max(confidence_matrix):
                            color = "green"
                        else:
                            color = "blue"
                        
                        pygame.draw.circle(
                            canvas,
                            color,
                            (
                                int(self.pix_square_size * x + self.pix_square_size / 2),
                                int(self.pix_square_size * y + self.pix_square_size / 2),
                            ),
                            int((self.pix_square_size / 2 - self.pix_square_size * 0.1) * confidence_amount),
                        )

        canvas_rect = canvas.get_rect()
        canvas_rect.center = background.get_rect().center
        background.blit(canvas, canvas_rect)

        if self.render_mode == "human":
            canvas_rect = canvas.get_rect()
            canvas_rect.center = self.window.get_rect().center
            self.window.blit(canvas, canvas_rect)
            pygame.event.pump()
            pygame.display.update()

            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(background)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def _setup_master_board(self, hit_pos):
        self.bomb_index = helper.generate_unique_coordinates(
            self.num_bombs, self.size[0] - 1, self.size[1] - 1, except_=hit_pos
        )

        self._master_board[*self.bomb_index] = Identifier.BOMB.value
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
            revealed = self._soft_reveal_cell(cell)
            if revealed == Identifier.NOTHING.value:
                reveal_all = True
            elif revealed == Identifier.BOMB.value:
                self.reward += -1
                self._terminated = True

            counter = 0
            for i in [[1, 0], [-1, 0], [0, 1], [0, -1]]:
                if helper.index_in_bound(cell + i, self._master_board.shape):
                    master_value = self._master_board[*cell + i]
                    if self._board[*cell + i] == Identifier.UNREVEALED:
                        counter += 1
                    if master_value == Identifier.NOTHING.value or reveal_all:
                        cells.append(cell + i)
                else:
                    counter += 1
            if counter == 4:
                self.reward += -0.02

            del cells[0]

    def _soft_reveal_cell(self, cell):
        value = self._master_board[*cell]
        self._board[*cell] = value
        return value

    def _check_win(self):
        unrevealed = sum(sum(self._board == Identifier.UNREVEALED.value))
        self.win = unrevealed == self.num_bombs
        return self.win

    def _update_mask(self):
        self._mask = np.zeros(self.action_space.n, dtype=bool)
        temp = self._board.reshape(1, -1)
        self._mask[temp[0] != Identifier.UNREVEALED.value] = True