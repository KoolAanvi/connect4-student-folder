from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .env import Connect4Config, Connect4Env
from .opponents import Agent


@dataclass(frozen=True)
class DQNConfig:
    gamma: float = ?
    learning_rate: float = ?
    batch_size: int = ?
    replay_capacity: int = ? #M for replay buffer capacity
    min_replay_size: int = ? #minimum replay buffer size before starting optimization
    target_sync_interval: int = ? #C for target network sync
    train_interval: int = ?     #N for number of environment steps between optimization updates for replay ratio
    hidden_dim: int = ?
    epsilon_start: float = ?
    epsilon_end: float = ?
    epsilon_decay_steps: int = ?
    max_episodes: int = ?
    eval_interval: int = 50
    eval_games: int = 40
    gradient_clip_norm: float = ?
    seed: int = 42

    def epsilon_at_step(self, step: int) -> float:
        if self.epsilon_decay_steps <= 0:
            return self.epsilon_end
        mix = min(max(step, 0) / self.epsilon_decay_steps, 1.0)
        return self.epsilon_start + mix * (self.epsilon_end - self.epsilon_start)


class ReplayBuffer:
    """
    Store DQN transitions and support random minibatch sampling.

    Inputs to `push` should be one transition:
    observation, action, reward, next observation, next legal-action mask, and done flag.
    `sample(batch_size)` should return batched NumPy arrays suitable for training.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        raise NotImplementedError("TODO: implement replay buffer storage")

    def __len__(self) -> int:
        raise NotImplementedError("TODO: return replay buffer size")

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        next_legal_mask: np.ndarray,
        done: bool,
    ) -> None:
        raise NotImplementedError("TODO: append one transition to the replay buffer")

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError("TODO: sample a minibatch of transitions")


class Connect4QNetwork(nn.Module):
    """
    Map an observation tensor to one Q-value per action.

    The forward pass should accept a tensor with shape `(batch_size, 2, rows, cols)`
    and return a tensor with shape `(batch_size, action_size)`.
    """

    def __init__(self, observation_shape: tuple[int, int, int], action_size: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.observation_shape = observation_shape
        self.action_size = action_size
        self.hidden_dim = hidden_dim
        raise NotImplementedError("TODO: define your Q-network layers")

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("TODO: return a tensor of shape (batch_size, action_size)")


def _masked_argmax(q_values: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """Mask illegal actions before taking argmax."""
    masked_q = q_values.masked_fill(~legal_mask, -1e9)
    return masked_q.argmax(dim=1)


def _select_action(
    model: Connect4QNetwork,
    obs: np.ndarray,
    legal_mask: np.ndarray,
    epsilon: float,
    device: torch.device,
    rng: np.random.Generator,
) -> int:
    """TODO: implement epsilon-greedy action selection with legal-action masking."""
    raise NotImplementedError("TODO: implement epsilon-greedy action selection")


def _apply_opponent_turn(env: Connect4Env, opponent: Agent) -> tuple[float, bool]:
    """Advance the environment through one scripted opponent move."""
    if env.done:
        raise RuntimeError("Cannot apply opponent turn to a finished game")
    action = opponent.select_action(env)
    _, _, done, info = env.step(action)
    if not done:
        return 0.0, False
    return float(info["opponent_reward"]), True


def _optimize_model(
    online_model: Connect4QNetwork,
    target_model: Connect4QNetwork,
    optimizer: torch.optim.Optimizer,
    replay_buffer: ReplayBuffer,
    config: DQNConfig,
    device: torch.device,
) -> float:
    """TODO: run one DQN optimization step and return the scalar loss."""
    raise NotImplementedError("TODO: implement one DQN update step")


class DQNPolicyAgent:
    """Minimal agent wrapper expected by evaluate.py and play_connect4.py."""

    def __init__(
        self,
        model: Connect4QNetwork,
        config: Connect4Config,
        device: torch.device | str | None = None,
        name: str = "dqn",
    ) -> None:
        self.model = model
        self.config = config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.name = name

    def select_action(self, env: Connect4Env) -> int:
        raise NotImplementedError("TODO: choose an action for evaluation using your trained model")

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path, device: torch.device | str | None = None) -> "DQNPolicyAgent":
        checkpoint = torch.load(checkpoint_path, map_location=device or "cpu")
        env_config = Connect4Config(**checkpoint["env_config"])
        hidden_dim = int(checkpoint["training_config"]["hidden_dim"])
        model = Connect4QNetwork(
            observation_shape=env_config.observation_shape,
            action_size=env_config.action_size,
            hidden_dim=hidden_dim,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        return cls(model=model, config=env_config, device=device)


def save_dqn_checkpoint(
    checkpoint_path: str | Path,
    model: Connect4QNetwork,
    env_config: Connect4Config,
    training_config: DQNConfig,
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


def train_dqn(
    env_config: Connect4Config,
    training_config: DQNConfig,
    opponent: Agent,
    eval_opponents: dict[str, Agent] | None = None,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    raise NotImplementedError("TODO: implement the DQN training loop")