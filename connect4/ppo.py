from __future__ import annotations

import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from .env import Connect4Config, Connect4Env
from .evaluate import evaluate_agent_pair
from .opponents import Agent, build_agent


def _repo_root() -> Path:
    """Student package root (parent of the `connect4` package)."""
    return Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    rollout_steps: int = 512        # steps collected per update
    update_epochs: int = 4          # gradient epochs per rollout
    minibatch_size: int = 128       # minibatch size for PPO updates
    clip_coef: float = 0.2          # PPO clipping parameter (epsilon)
    value_coef: float = 0.5         # weight on value loss
    entropy_coef: float = 0.01      # entropy bonus weight
    max_grad_norm: float = 0.5      # gradient clipping
    hidden_dim: int = 256           # hidden units in MLP heads
    max_updates: int = 1000         # total number of PPO updates to run
    eval_interval: int = 20         # evaluate every N updates
    eval_games: int = 100           # match homework-style eval (e.g. 100 games)
    seed: int = 42

    def epsilon_at_step(self, step: int) -> float:
        # Not used for PPO, kept for API compatibility
        return 0.0


class Connect4ActorCritic(nn.Module):
    """
    Shared CNN backbone with separate policy (actor) and value (critic) heads.

    Input:  (batch, 2, rows, cols)
    Output: (policy_logits: (batch, action_size),
             state_values:  (batch,))
    """

    def __init__(
        self,
        observation_shape: tuple[int, int, int],
        action_size: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.observation_shape = observation_shape
        self.action_size = action_size
        self.hidden_dim = hidden_dim

        in_channels, rows, cols = observation_shape

        # Shared convolutional backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 64,  kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64,          128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128,         128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        flat_dim = 128 * rows * cols  # 128 * 6 * 7 = 5376 for default board

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_size),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(obs)
        logits = self.policy_head(features)
        values = self.value_head(features).squeeze(-1)  # (batch,)
        return logits, values


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
    """
    After a reset the opponent may move first (when start_player == -1).
    Step the opponent until it is the learning agent's turn (player == 1).
    Returns True if the game ended during the opponent's turn.
    """
    while env.current_player != 1:
        if env.done:
            return True
        _, done = _apply_opponent_turn(env, opponent)
        if done:
            return True
    return False


def _obs_to_tensor(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(obs).unsqueeze(0).to(device=device, dtype=torch.float32)


def _mask_to_tensor(mask: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(mask).unsqueeze(0).to(device=device, dtype=torch.bool)


def _sample_action(
    model: Connect4ActorCritic,
    obs: np.ndarray,
    legal_mask: np.ndarray,
    device: torch.device,
) -> tuple[int, float, float]:
    """
    Sample an action stochastically from the masked policy.
    Returns (action, log_prob, value).
    """
    obs_t  = _obs_to_tensor(obs,        device)
    mask_t = _mask_to_tensor(legal_mask, device)

    with torch.no_grad():
        logits, value = model(obs_t)
        dist   = _masked_distribution(logits, mask_t)
        action = dist.sample()
        log_prob = dist.log_prob(action)

    return int(action.item()), float(log_prob.item()), float(value.item())


def _greedy_action(
    model: Connect4ActorCritic,
    obs: np.ndarray,
    legal_mask: np.ndarray,
    device: torch.device,
) -> int:
    """Choose the highest-probability legal action (greedy / eval-time)."""
    obs_t  = _obs_to_tensor(obs,        device)
    mask_t = _mask_to_tensor(legal_mask, device)

    with torch.no_grad():
        logits, _ = model(obs_t)
        action = _masked_distribution(logits, mask_t).probs.argmax(dim=1)

    return int(action.item())


def _bootstrap_value(
    model: Connect4ActorCritic,
    obs: np.ndarray,
    device: torch.device,
) -> float:
    """Estimate V(s) for the final state in a rollout."""
    obs_t = _obs_to_tensor(obs, device)
    with torch.no_grad():
        _, value = model(obs_t)
    return float(value.item())


def _compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    config: PPOConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Generalized Advantage Estimates (GAE) and discounted returns.
    Returns (advantages, returns), both shape (T,).
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0

    for t in reversed(range(T)):
        next_val          = last_value if t == T - 1 else values[t + 1]
        next_non_terminal = 1.0 - (dones[t + 1] if t < T - 1 else float(dones[-1]))
        delta    = rewards[t] + config.gamma * next_val * next_non_terminal - values[t]
        last_gae = delta + config.gamma * config.gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


class PPOPolicyAgent:
    """Minimal agent wrapper expected by evaluate.py and play_connect4.py."""

    def __init__(
        self,
        model: Connect4ActorCritic,
        config: Connect4Config,
        device: torch.device | str | None = None,
        name: str = "ppo",
    ) -> None:
        self.model  = model
        self.config = config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.name   = name
        self.model.to(self.device)
        self.model.eval()

    def select_action(self, env: Connect4Env) -> int:
        """Greedy action selection used during evaluation / play."""
        return _greedy_action(
            model=self.model,
            obs=env.get_observation(),
            legal_mask=env.legal_actions_mask(),
            device=self.device,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: torch.device | str | None = None,
    ) -> "PPOPolicyAgent":
        checkpoint = torch.load(checkpoint_path, map_location=device or "cpu")
        env_config  = Connect4Config(**checkpoint["env_config"])
        hidden_dim  = int(checkpoint["training_config"]["hidden_dim"])
        model = Connect4ActorCritic(
            observation_shape=env_config.observation_shape,
            action_size=env_config.action_size,
            hidden_dim=hidden_dim,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        return cls(model=model, config=env_config, device=device)


def _write_ppo_history_csv(history: list[tuple[int, float, float, int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "update",
                "win_rate_vs_training_opponent",
                "avg_episode_reward_last200",
                "episodes_completed_at_eval",
            ]
        )
        writer.writerows(history)


def _write_episode_rewards_csv(episode_rewards: list[float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward"])
        for idx, reward in enumerate(episode_rewards, start=1):
            writer.writerow([idx, reward])


def _moving_average(values: list[float], window: int) -> np.ndarray:
    if not values:
        return np.array([], dtype=np.float32)
    if window <= 1:
        return np.asarray(values, dtype=np.float32)
    arr = np.asarray(values, dtype=np.float32)
    out = np.zeros_like(arr)
    cumsum = np.cumsum(arr)
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        total = cumsum[i] - (cumsum[start - 1] if start > 0 else 0.0)
        out[i] = total / float(i - start + 1)
    return out


def _plot_ppo_training_progress(
    episode_rewards: list[float],
    history: list[tuple[int, float, float, int]],
    path: Path,
) -> None:
    """Save a DQN-style plot: rewards, moving average, and eval win-rate points."""
    if not episode_rewards:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    episodes = np.arange(1, len(episode_rewards) + 1)
    moving_avg = _moving_average(episode_rewards, window=50)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(episodes, episode_rewards, color="tab:blue", alpha=0.25, label="Episode reward")
    ax.plot(episodes, moving_avg, color="tab:orange", linewidth=1.8, label="50-episode moving average")

    if history:
        eval_episode_idx = [h[3] for h in history]
        eval_win_rates = [h[1] for h in history]
        ax.plot(
            eval_episode_idx,
            eval_win_rates,
            color="tab:green",
            marker="o",
            linewidth=1.8,
            label="Eval win rate vs random",
        )

    ax.axhline(0.8, color="gray", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_title("PPO Connect 4 Training Progress")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward / win rate")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


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
        "env_config":      asdict(env_config),
        "training_config": asdict(training_config),
        "model_state_dict": model.state_dict(),
        "metadata":        metadata or {},
    }
    torch.save(payload, path)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_ppo(
    env_config: Connect4Config,
    training_config: PPOConfig,
    opponent: Agent,
    eval_opponents: dict[str, Agent] | None = None,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """
    Full PPO training loop against a scripted opponent.

    The learning agent always plays as player 1.
    Games alternate which side moves first so the agent learns both positions.

    Returns a dict with keys:
        model              – trained Connect4ActorCritic
        history            – list of (update_idx, win_rate, avg_ep_reward, episodes_done) checkpoints
        episode_rewards    – list of per-episode total rewards
        device             – torch.device used
        training_loop_s    – wall seconds for the main PPO update loop
        total_wall_s       – wall seconds including final eval and artifact I/O
        final_win_rate_100 – player-one win rate in a fresh 100-game eval vs random
        history_csv        – path to written CSV (if any history)
        episode_rewards_csv – path to per-episode rewards CSV (if any episodes)
        training_plot      – path to saved matplotlib figure (if matplotlib available)
    """
    cfg     = training_config
    device_ = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    model     = Connect4ActorCritic(
        observation_shape=env_config.observation_shape,
        action_size=env_config.action_size,
        hidden_dim=cfg.hidden_dim,
    ).to(device_)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, eps=1e-5)

    # ── Pre-allocate rollout storage ────────────────────────────────────────
    T          = cfg.rollout_steps
    obs_shape  = env_config.observation_shape
    n_actions  = env_config.action_size

    buf_obs      = np.zeros((T, *obs_shape),    dtype=np.float32)
    buf_actions  = np.zeros(T,                   dtype=np.int64)
    buf_log_probs= np.zeros(T,                   dtype=np.float32)
    buf_rewards  = np.zeros(T,                   dtype=np.float32)
    buf_values   = np.zeros(T,                   dtype=np.float32)
    buf_dones    = np.zeros(T,                   dtype=np.float32)
    buf_masks    = np.zeros((T, n_actions),      dtype=bool)

    history         = []
    episode_rewards = []
    ep_reward       = 0.0
    game_idx        = 0

    # Start first game; alternate who goes first
    env = Connect4Env(env_config)
    start_player = 1
    obs, _ = env.reset(start_player=start_player)
    if _advance_to_agent_turn(env, opponent):
        obs, _ = env.reset(start_player=start_player)

    print(f"PPO training on {device_}. "
          f"{cfg.max_updates} updates × {cfg.rollout_steps} steps = "
          f"{cfg.max_updates * cfg.rollout_steps:,} total steps.")

    t_loop_start = time.perf_counter()
    for update_idx in range(cfg.max_updates):
        model.eval()

        # ── Collect rollout ─────────────────────────────────────────────────
        for t in range(T):
            legal_mask = env.legal_actions_mask()
            action, log_prob, value = _sample_action(model, obs, legal_mask, device_)

            buf_obs[t]       = obs
            buf_actions[t]   = action
            buf_log_probs[t] = log_prob
            buf_values[t]    = value
            buf_masks[t]     = legal_mask

            next_obs, reward, done, info = env.step(action)
            ep_reward += reward

            if done:
                buf_rewards[t] = reward
                buf_dones[t]   = 1.0
                episode_rewards.append(ep_reward)
                ep_reward = 0.0

                # Start next game
                game_idx    += 1
                start_player = 1 if game_idx % 2 == 0 else -1
                obs, _       = env.reset(start_player=start_player)
                game_ended   = _advance_to_agent_turn(env, opponent)
                if game_ended:
                    # Opponent won immediately; loop will start fresh next step
                    obs, _ = env.reset(start_player=1)
            else:
                # Let opponent move, collect its outcome reward for our agent
                opp_reward, opp_done = _apply_opponent_turn(env, opponent)
                step_reward = reward + opp_reward   # usually reward == 0 here
                buf_rewards[t] = step_reward
                ep_reward     += opp_reward

                if opp_done:
                    buf_dones[t]   = 1.0
                    episode_rewards.append(ep_reward)
                    ep_reward = 0.0
                    game_idx    += 1
                    start_player = 1 if game_idx % 2 == 0 else -1
                    obs, _       = env.reset(start_player=start_player)
                    _advance_to_agent_turn(env, opponent)
                else:
                    buf_dones[t] = 0.0
                    obs = env.get_observation()

        # Bootstrap value for the last state
        last_value = _bootstrap_value(model, obs, device_) if not env.done else 0.0

        # ── GAE ─────────────────────────────────────────────────────────────
        advantages, returns = _compute_gae(
            buf_rewards, buf_values, buf_dones, last_value, cfg
        )
        adv_mean = advantages.mean()
        adv_std  = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        # Convert buffers to tensors
        t_obs      = torch.from_numpy(buf_obs).to(device_)
        t_actions  = torch.from_numpy(buf_actions).to(device_)
        t_log_old  = torch.from_numpy(buf_log_probs).to(device_)
        t_adv      = torch.from_numpy(advantages).to(device_)
        t_returns  = torch.from_numpy(returns).to(device_)
        t_masks    = torch.from_numpy(buf_masks).to(device_)

        # ── PPO update ───────────────────────────────────────────────────────
        model.train()
        indices = np.arange(T)

        for _ in range(cfg.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, T, cfg.minibatch_size):
                mb = indices[start:start + cfg.minibatch_size]

                mb_obs     = t_obs[mb]
                mb_actions = t_actions[mb]
                mb_log_old = t_log_old[mb]
                mb_adv     = t_adv[mb]
                mb_returns = t_returns[mb]
                mb_masks   = t_masks[mb]

                logits, values_pred = model(mb_obs)
                dist     = _masked_distribution(logits, mb_masks)
                new_lp   = dist.log_prob(mb_actions)
                entropy  = dist.entropy()

                ratio    = (new_lp - mb_log_old).exp()
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * ratio.clamp(1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss  = torch.max(pg_loss1, pg_loss2).mean()

                v_loss   = 0.5 * (values_pred - mb_returns).pow(2).mean()
                ent_loss = entropy.mean()

                loss = pg_loss + cfg.value_coef * v_loss - cfg.entropy_coef * ent_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()

        # ── Periodic evaluation ──────────────────────────────────────────────
        if (update_idx + 1) % cfg.eval_interval == 0:
            model.eval()
            agent      = PPOPolicyAgent(model, env_config, device_)
            opp_name   = getattr(opponent, "name", "opponent")
            results    = evaluate_agent_pair(agent, opponent, games=cfg.eval_games, config=env_config)
            win_rate   = results["player_one_win_rate"]
            avg_rew    = float(np.mean(episode_rewards[-200:])) if episode_rewards else 0.0
            history.append((update_idx + 1, win_rate, avg_rew, len(episode_rewards)))
            print(
                f"Update {update_idx+1:>5} | WinRate vs {opp_name}: {win_rate:.1%} "
                f"| AvgEpReward: {avg_rew:.3f}"
            )

            # Additional eval opponents (e.g. heuristic)
            if eval_opponents:
                for name, opp in eval_opponents.items():
                    r = evaluate_agent_pair(agent, opp, games=cfg.eval_games, config=env_config)
                    print(f"           WinRate vs {name}: {r['player_one_win_rate']:.1%}")

    training_loop_s = time.perf_counter() - t_loop_start

    # ── Final 100-game eval vs fresh random (homework-style report) ─────────
    model.eval()
    trained_agent = PPOPolicyAgent(model, env_config, device_)
    random_fresh = build_agent("random", seed=cfg.seed + 99_999)
    final_eval = evaluate_agent_pair(
        trained_agent,
        random_fresh,
        games=100,
        config=env_config,
    )
    final_win_rate_100 = float(final_eval["player_one_win_rate"])

    root = _repo_root()
    history_csv_path = root / "checkpoints" / "ppo_training_history.csv"
    episode_rewards_csv_path = root / "checkpoints" / "ppo_episode_rewards.csv"
    plot_saved: Path | None = None

    if episode_rewards:
        _write_episode_rewards_csv(episode_rewards, episode_rewards_csv_path)
        print(f"Episode rewards CSV → {episode_rewards_csv_path}")

    if history:
        _write_ppo_history_csv(history, history_csv_path)
        print(f"Training history CSV → {history_csv_path}")
        plot_out = root / "plots" / "ppo_training_progress.png"
        try:
            _plot_ppo_training_progress(episode_rewards, history, plot_out)
            plot_saved = plot_out
            print(f"Training plot → {plot_saved}")
        except ImportError:
            print("matplotlib not installed; skipped training plot. Install with: pip install matplotlib")

    # ── Final save ───────────────────────────────────────────────────────────
    save_ppo_checkpoint(
        "checkpoints/connect4_ppo.pt",
        model,
        env_config,
        training_config,
        metadata={
            "final_win_rate_last_periodic_eval": history[-1][1] if history else None,
            "final_win_rate_100_games_vs_random": final_win_rate_100,
            "training_loop_wall_s": training_loop_s,
        },
    )
    print("Checkpoint saved → checkpoints/connect4_ppo.pt")

    total_wall_s = time.perf_counter() - t_loop_start
    print(
        f"Final eval (100 games vs random): {final_win_rate_100:.1%} "
        f"(wins={final_eval['player_one_wins']}, losses={final_eval['player_two_wins']}, "
        f"draws={final_eval['draws']})"
    )
    print(f"PPO update loop wall time: {training_loop_s:.1f}s")
    print(f"Total wall time (loop + final eval + CSV/plot/checkpoint): {total_wall_s:.1f}s")

    return {
        "model":                model,
        "history":              history,
        "episode_rewards":      episode_rewards,
        "device":               device_,
        "training_loop_s":      training_loop_s,
        "total_wall_s":         total_wall_s,
        "final_win_rate_100":   final_win_rate_100,
        "history_csv":          str(history_csv_path) if history else None,
        "episode_rewards_csv":  str(episode_rewards_csv_path) if episode_rewards else None,
        "training_plot":        str(plot_saved) if plot_saved is not None else None,
    }