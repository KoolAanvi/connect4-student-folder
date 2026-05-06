from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .env import Connect4Config, Connect4Env


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(channels)
        self.norm2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return F.relu(x + residual)


class StrongConnect4QNetwork(nn.Module):
    def __init__(
        self,
        observation_shape: tuple[int, int, int],
        action_size: int,
        conv_channels: int = 128,
        residual_blocks: int = 4,
    ) -> None:
        super().__init__()
        channels, _, _ = observation_shape
        self.stem = nn.Sequential(
            nn.Conv2d(channels, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(conv_channels) for _ in range(residual_blocks)])
        self.column_adv = nn.Sequential(
            nn.Linear(conv_channels, conv_channels),
            nn.ReLU(),
            nn.Linear(conv_channels, 1),
        )
        self.state_value = nn.Sequential(
            nn.Linear(conv_channels, conv_channels),
            nn.ReLU(),
            nn.Linear(conv_channels, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.blocks(self.stem(obs))
        global_feat = x.mean(dim=(2, 3))
        column_feat = x.mean(dim=2).transpose(1, 2)
        adv = self.column_adv(column_feat).squeeze(-1)
        value = self.state_value(global_feat)
        return value + adv - adv.mean(dim=1, keepdim=True)


def _masked_argmax(q_values: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    masked_q = q_values.masked_fill(~legal_mask, -1e9)
    return masked_q.argmax(dim=1)


def _select_action(
    model: StrongConnect4QNetwork,
    obs: np.ndarray,
    legal_mask: np.ndarray,
    device: torch.device,
) -> int:
    obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device=device, dtype=torch.float32)
    legal_tensor = torch.from_numpy(legal_mask).unsqueeze(0).to(device=device, dtype=torch.bool)
    with torch.no_grad():
        q_values = model(obs_tensor)
        action = _masked_argmax(q_values, legal_tensor)
    return int(action.item())


class StrongDQNPolicyAgent:
    def __init__(
        self,
        model: StrongConnect4QNetwork,
        config: Connect4Config,
        device: torch.device | str | None = None,
        name: str = "strong-dqn",
    ) -> None:
        self.model = model
        self.config = config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.name = name
        self.model.to(self.device)
        self.model.eval()

    def select_action(self, env: Connect4Env) -> int:
        if env.config != self.config:
            raise ValueError("Environment config does not match the loaded strong DQN checkpoint")
        return _select_action(
            model=self.model,
            obs=env.get_observation(),
            legal_mask=env.legal_actions_mask(),
            device=self.device,
        )

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path, device: torch.device | str | None = None) -> "StrongDQNPolicyAgent":
        checkpoint = torch.load(checkpoint_path, map_location=device or "cpu")
        env_config = Connect4Config(**checkpoint["env_config"])
        training_config = checkpoint.get("training_config", {})
        model = StrongConnect4QNetwork(
            observation_shape=env_config.observation_shape,
            action_size=env_config.action_size,
            conv_channels=int(training_config.get("conv_channels", 128)),
            residual_blocks=int(training_config.get("residual_blocks", 4)),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        return cls(model=model, config=env_config, device=device)