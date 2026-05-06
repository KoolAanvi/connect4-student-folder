from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np

from .env import Connect4Env

if TYPE_CHECKING:
    from .strong_dqn import StrongDQNPolicyAgent


FINAL_BOSS_CHECKPOINT_CANDIDATES = [
    Path(__file__).resolve().parent.parent / "checkpoints" / "finalboss.pt",
]


class Agent(Protocol):
    name: str

    def select_action(self, env: Connect4Env) -> int:
        ...


@dataclass
class RandomAgent:
    seed: int | None = None
    name: str = "random"
    rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def select_action(self, env: Connect4Env) -> int:
        legal_actions = env.legal_actions()
        if not legal_actions:
            raise RuntimeError("No legal actions available")
        return int(self.rng.choice(legal_actions))


@dataclass
class HeuristicAgent:
    seed: int | None = None
    name: str = "heuristic"
    rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def select_action(self, env: Connect4Env) -> int:
        legal_actions = env.legal_actions()
        if not legal_actions:
            raise RuntimeError("No legal actions available")

        winning_moves = env.winning_actions(env.current_player)
        if winning_moves:
            return self._choose_center_preferred(winning_moves, env.config.cols)

        blocking_moves = [
            action for action in legal_actions if action in env.winning_actions(-env.current_player)
        ]
        if blocking_moves:
            return self._choose_center_preferred(blocking_moves, env.config.cols)

        return self._choose_center_preferred(legal_actions, env.config.cols)

    def _choose_center_preferred(self, actions: list[int], cols: int) -> int:
        center = (cols - 1) / 2.0
        distances = np.array([abs(action - center) for action in actions], dtype=np.float32)
        min_distance = float(distances.min())
        best_actions = [action for action, distance in zip(actions, distances) if distance == min_distance]
        return int(self.rng.choice(best_actions))


def load_final_boss() -> Agent:
    from .strong_dqn import StrongDQNPolicyAgent

    checkpoint_path = next((path for path in FINAL_BOSS_CHECKPOINT_CANDIDATES if path.exists()), None)
    if checkpoint_path is None:
        searched = ", ".join(str(path) for path in FINAL_BOSS_CHECKPOINT_CANDIDATES)
        raise ValueError(
            f"The Final Boss checkpoint was not found. Looked for: {searched}. "
            "Make sure checkpoints/finalboss.pt is included with the student folder."
        )
    boss = StrongDQNPolicyAgent.from_checkpoint(checkpoint_path)
    boss.name = "The Final Boss"
    return boss


def build_agent(agent_name: str, seed: int | None = None) -> Agent:
    normalized = agent_name.lower()
    if normalized == "random":
        return RandomAgent(seed=seed)
    if normalized == "heuristic":
        return HeuristicAgent(seed=seed)
    if normalized in {"final-boss", "the-final-boss", "boss"}:
        return load_final_boss()
    raise ValueError(f"Unsupported agent type: {agent_name}")