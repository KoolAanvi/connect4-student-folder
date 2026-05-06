#!/usr/bin/env python3
"""CLI entry point for PPO training on Connect 4."""

from __future__ import annotations

import argparse
from dataclasses import replace

from connect4.env import Connect4Config
from connect4.opponents import HeuristicAgent, build_agent
from connect4.ppo import PPOConfig, train_ppo


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a PPO agent for Connect 4.")
    parser.add_argument(
        "--opponent",
        default="random",
        choices=("random", "heuristic"),
        help="Scripted opponent during self-play-style training.",
    )
    parser.add_argument("--seed", type=int, default=None, help="RNG seed (default: PPOConfig.seed).")
    parser.add_argument("--max-updates", type=int, default=None, help="Override PPOConfig.max_updates.")
    parser.add_argument("--device", default=None, help="Torch device, e.g. cuda or cpu.")
    args = parser.parse_args()

    env_config = Connect4Config()
    training_config = PPOConfig()
    if args.seed is not None:
        training_config = replace(training_config, seed=args.seed)
    if args.max_updates is not None:
        training_config = replace(training_config, max_updates=args.max_updates)

    opponent = build_agent(args.opponent, seed=args.seed)
    eval_opponents = {"heuristic": HeuristicAgent(seed=args.seed)} if args.opponent == "random" else None

    train_ppo(
        env_config,
        training_config,
        opponent,
        eval_opponents=eval_opponents,
        device=args.device,
    )


if __name__ == "__main__":
    main()
