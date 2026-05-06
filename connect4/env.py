from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Connect4Config:
    rows: int = 6
    cols: int = 7
    connect_n: int = 4
    connect_two_reward: float = 0.05
    connect_three_reward: float = 0.10
    win_reward: float = 1.0
    loss_reward: float = -1.0
    draw_reward: float = 0.0
    invalid_action_reward: float = -1.0

    def __post_init__(self) -> None:
        if self.rows < 4 or self.cols < 4:
            raise ValueError("rows and cols must both be at least 4")
        if self.connect_n < 3:
            raise ValueError("connect_n must be at least 3")
        if self.connect_n > max(self.rows, self.cols):
            raise ValueError("connect_n cannot exceed the larger board dimension")

    @property
    def observation_shape(self) -> tuple[int, int, int]:
        return (2, self.rows, self.cols)

    @property
    def action_size(self) -> int:
        return self.cols

    def reward_for_connection(self, line_length: int) -> float:
        if line_length >= self.connect_n:
            return float(self.win_reward)
        if line_length >= 3:
            return float(self.connect_three_reward)
        if line_length == 2:
            return float(self.connect_two_reward)
        return 0.0


class Connect4Env:
    def __init__(self, config: Connect4Config | None = None) -> None:
        self.config = config or Connect4Config()
        self.board = np.zeros((self.config.rows, self.config.cols), dtype=np.int8)
        self.current_player = 1
        self.winner: int | None = None
        self.done = False
        self.move_count = 0
        self.last_action: tuple[int, int] | None = None
        self.last_connection_length = 0

    def copy(self) -> "Connect4Env":
        env = Connect4Env(self.config)
        env.board = self.board.copy()
        env.current_player = self.current_player
        env.winner = self.winner
        env.done = self.done
        env.move_count = self.move_count
        env.last_action = self.last_action
        env.last_connection_length = self.last_connection_length
        return env

    def reset(self, start_player: int = 1) -> tuple[np.ndarray, dict[str, object]]:
        if start_player not in (-1, 1):
            raise ValueError("start_player must be either 1 or -1")
        self.board.fill(0)
        self.current_player = start_player
        self.winner = None
        self.done = False
        self.move_count = 0
        self.last_action = None
        self.last_connection_length = 0
        return self.get_observation(), self._build_info(acting_player=None, invalid_action=False)

    def raw_board(self) -> np.ndarray:
        return self.board.copy()

    def get_observation(self, player: int | None = None) -> np.ndarray:
        perspective_player = self.current_player if player is None else player
        if perspective_player not in (-1, 1):
            raise ValueError("player must be either 1 or -1")
        own_plane = (self.board == perspective_player).astype(np.float32)
        opp_plane = (self.board == -perspective_player).astype(np.float32)
        return np.stack([own_plane, opp_plane], axis=0)

    def legal_actions(self) -> list[int]:
        return [col for col in range(self.config.cols) if self.board[0, col] == 0]

    def legal_actions_mask(self) -> np.ndarray:
        return self.board[0] == 0

    def is_action_legal(self, action: int) -> bool:
        return 0 <= action < self.config.cols and self.board[0, action] == 0

    def winning_actions(self, player: int | None = None) -> list[int]:
        target_player = self.current_player if player is None else player
        if target_player not in (-1, 1):
            raise ValueError("player must be either 1 or -1")
        wins: list[int] = []
        for action in self.legal_actions():
            if self._would_win(action, target_player):
                wins.append(action)
        return wins

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, object]]:
        if self.done:
            raise RuntimeError("Cannot call step() on a finished game. Call reset() first.")

        acting_player = self.current_player
        if not self.is_action_legal(action):
            self.done = True
            self.winner = -acting_player
            reward = float(self.config.invalid_action_reward)
            opponent_reward = float(self.config.win_reward)
            return self.get_observation(), reward, True, self._build_info(
                acting_player=acting_player,
                invalid_action=True,
                opponent_reward=opponent_reward,
            )

        row = self._find_drop_row(action)
        if row is None:
            raise RuntimeError("Internal error: legal action had no available slot")

        self.board[row, action] = acting_player
        self.move_count += 1
        self.last_action = (row, action)
        self.last_connection_length = self._max_connection_length(row, action, acting_player)

        if self.last_connection_length >= self.config.connect_n:
            self.done = True
            self.winner = acting_player
            reward = float(self.config.win_reward)
            opponent_reward = float(self.config.loss_reward)
        elif self.move_count == self.config.rows * self.config.cols:
            self.done = True
            self.winner = 0
            reward = float(self.config.draw_reward)
            opponent_reward = float(self.config.draw_reward)
        else:
            self.current_player = -acting_player
            reward = self.config.reward_for_connection(self.last_connection_length)
            opponent_reward = 0.0

        return self.get_observation(), reward, self.done, self._build_info(
            acting_player=acting_player,
            invalid_action=False,
            opponent_reward=opponent_reward,
        )

    def render(self) -> str:
        symbols = {1: "X", -1: "O", 0: "."}
        header = "  " + " ".join(str(col) for col in range(self.config.cols))
        lines = [header]
        for row in range(self.config.rows):
            tokens = " ".join(symbols[int(cell)] for cell in self.board[row])
            lines.append(f"{row} {tokens}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.render()

    def _build_info(self, acting_player: int | None, invalid_action: bool, opponent_reward: float = 0.0) -> dict[str, object]:
        return {
            "acting_player": acting_player,
            "current_player": self.current_player,
            "winner": self.winner,
            "invalid_action": invalid_action,
            "opponent_reward": float(opponent_reward),
            "is_draw": self.done and self.winner == 0,
            "last_action": self.last_action,
            "last_connection_length": self.last_connection_length,
            "move_count": self.move_count,
            "legal_actions_mask": self.legal_actions_mask().copy(),
        }

    def _find_drop_row(self, action: int) -> int | None:
        column = self.board[:, action]
        available = np.flatnonzero(column == 0)
        if available.size == 0:
            return None
        return int(available[-1])

    def _would_win(self, action: int, player: int) -> bool:
        row = self._find_drop_row(action)
        if row is None:
            return False
        self.board[row, action] = player
        is_win = self._max_connection_length(row, action, player) >= self.config.connect_n
        self.board[row, action] = 0
        return is_win

    def _count_direction(self, row: int, col: int, delta_row: int, delta_col: int, player: int) -> int:
        count = 0
        row += delta_row
        col += delta_col
        while 0 <= row < self.config.rows and 0 <= col < self.config.cols and self.board[row, col] == player:
            count += 1
            row += delta_row
            col += delta_col
        return count

    def _max_connection_length(self, row: int, col: int, player: int) -> int:
        directions = ((1, 0), (0, 1), (1, 1), (1, -1))
        best = 1
        for delta_row, delta_col in directions:
            total = 1
            total += self._count_direction(row, col, delta_row, delta_col, player)
            total += self._count_direction(row, col, -delta_row, -delta_col, player)
            best = max(best, total)
        return best