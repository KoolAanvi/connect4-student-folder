from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from .env import Connect4Config, Connect4Env
from .opponents import Agent


@dataclass(frozen=True)
class PPOConfig:
    gamma: float = ?
    gae_lambda: float = ?
    learning_rate: float = ?
    rollout_steps: int = ?
    update_epochs: int = ?
    minibatch_size: int = ?
    clip_coef: float = ?
    value_coef: float = ?
    entropy_coef: float = ?
    max_grad_norm: float = ?
    hidden_dim: int = ?
    max_updates: int = ?
    eval_interval: int = 20
    eval_games: int = 40
    seed: int = 42


class Connect4ActorCritic(nn.Module):
    """
    Map an observation tensor to policy logits and a state-value estimate.

    The forward pass should accept a tensor with shape `(batch_size, 2, rows, cols)`.
    It should return a pair `(policy_logits, state_values)`, where the shapes are
    `(batch_size, action_size)` and `(batch_size,)` respectively.
    """

    def __init__(self, observation_shape: tuple[int, int, int], action_size: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.observation_shape = observation_shape
        self.action_size = action_size
        self.hidden_dim = hidden_dim
        raise NotImplementedError("TODO: define your PPO policy/value network layers")

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("TODO: return (policy_logits, state_values)")


def _masked_distribution(logits: torch.Tensor, legal_mask: torch.Tensor) -> Categorical:
    """Mask illegal actions before building a categorical policy."""
    masked_logits = logits.masked_fill(~legal_mask, -1e9)
    return Categorical(logits=masked_logits)


def _apply_opponent_turn(env: Connect4Env, opponent: Agent) -> tuple[float, bool]:
    """Advance the environment through one scripted opponent move."""
    if env.done:
        raise RuntimeError("Cannot apply opponent turn to a finished game")
    action = opponent.select_action(env)
    _, _, done, info = env.step(action)
    if not done:
        return 0.0, False
    return float(info["opponent_reward"]), True


def _advance_to_agent_turn(env: Connect4Env, opponent: Agent) -> bool:
    """TODO: keep stepping until it is the learning agent's turn again."""
    raise NotImplementedError("TODO: advance past the opponent turn when needed")


def _sample_action(
    model: Connect4ActorCritic,
    obs: np.ndarray,
    legal_mask: np.ndarray,
    device: torch.device,
) -> tuple[int, float, float]:
    """TODO: sample an action and return action, log-probability, and value."""
    raise NotImplementedError("TODO: implement PPO action sampling")


def _greedy_action(
    model: Connect4ActorCritic,
    obs: np.ndarray,
    legal_mask: np.ndarray,
    device: torch.device,
) -> int:
    """TODO: choose the evaluation-time action from the policy."""
    raise NotImplementedError("TODO: implement greedy evaluation action selection")


def _bootstrap_value(
    model: Connect4ActorCritic,
    obs: np.ndarray,
    device: torch.device,
) -> float:
    """TODO: estimate the value of the final state in a rollout."""
    raise NotImplementedError("TODO: implement bootstrap value computation")


def _compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    config: PPOConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """TODO: compute generalized advantage estimates and returns."""
    raise NotImplementedError("TODO: implement GAE")


class PPOPolicyAgent:
    """Minimal agent wrapper expected by evaluate.py and play_connect4.py."""

    def __init__(
        self,
        model: Connect4ActorCritic,
        config: Connect4Config,
        device: torch.device | str | None = None,
        name: str = "ppo",
    ) -> None:
        self.model = model
        self.config = config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.name = name

    def select_action(self, env: Connect4Env) -> int:
        raise NotImplementedError("TODO: choose an action for evaluation using your trained model")

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path, device: torch.device | str | None = None) -> "PPOPolicyAgent":
        checkpoint = torch.load(checkpoint_path, map_location=device or "cpu")
        env_config = Connect4Config(**checkpoint["env_config"])
        hidden_dim = int(checkpoint["training_config"]["hidden_dim"])
        model = Connect4ActorCritic(
            observation_shape=env_config.observation_shape,
            action_size=env_config.action_size,
            hidden_dim=hidden_dim,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        return cls(model=model, config=env_config, device=device)


def save_ppo_checkpoint(
    checkpoint_path: str | Path,
    model: Connect4ActorCritic,
    env_config: Connect4Config,
    training_config: PPOConfig,
    metadata: dict[str, Any] | None = None,
) -> None:
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "env_config": asdict(env_config),
        "training_config": asdict(training_config),
        "model_state_dict": model.state_dict(),
        "metadata": metadata or {},
    }
    torch.save(payload, path)


def train_ppo(
    env_config: Connect4Config,
    training_config: PPOConfig,
    opponent: Agent,
    eval_opponents: dict[str, Agent] | None = None,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    raise NotImplementedError("TODO: implement the PPO training loop")