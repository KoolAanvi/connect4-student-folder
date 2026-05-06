from __future__ import annotations

import argparse

from connect4.env import Connect4Config
from connect4.evaluate import evaluate_agent_pair
from connect4.opponents import build_agent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark two scripted Connect-style agents.")
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--cols", type=int, default=7)
    parser.add_argument("--connect-n", type=int, default=4)
    parser.add_argument("--player1", choices=["random", "heuristic", "final-boss"], default="heuristic")
    parser.add_argument("--player2", choices=["random", "heuristic", "final-boss"], default="random")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-alternate-start", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    player_one = build_agent(args.player1, seed=args.seed)
    player_two = build_agent(args.player2, seed=args.seed + 1)

    config = getattr(player_one, "config", None)
    if not isinstance(config, Connect4Config):
        config = getattr(player_two, "config", None)
    if not isinstance(config, Connect4Config):
        config = Connect4Config(rows=args.rows, cols=args.cols, connect_n=args.connect_n)

    results = evaluate_agent_pair(
        player_one=player_one,
        player_two=player_two,
        games=args.games,
        config=config,
        alternate_start=not args.no_alternate_start,
    )

    print(f"Board: {args.rows}x{args.cols}, connect {args.connect_n}")
    print(f"Player 1: {player_one.name}")
    print(f"Player 2: {player_two.name}")
    print(f"Games: {results['games']}")
    print(f"Player 1 wins: {results['player_one_wins']} ({results['player_one_win_rate']:.3f})")
    print(f"Player 2 wins: {results['player_two_wins']} ({results['player_two_win_rate']:.3f})")
    print(f"Draws: {results['draws']} ({results['draw_rate']:.3f})")


if __name__ == "__main__":
    main()