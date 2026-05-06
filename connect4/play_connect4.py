from __future__ import annotations

import argparse
from importlib import import_module

from connect4.env import Connect4Config, Connect4Env
from connect4.opponents import Agent, build_agent


class HumanAgent:
    def __init__(self, name: str = "human") -> None:
        self.name = name

    def select_action(self, env: Connect4Env) -> int:
        legal_actions = env.legal_actions()
        prompt = f"Choose a column from {legal_actions} (or q to quit): "
        while True:
            response = input(prompt).strip().lower()
            if response in {"q", "quit", "exit"}:
                raise KeyboardInterrupt
            try:
                action = int(response)
            except ValueError:
                print("Please enter a valid integer column index.")
                continue
            if action not in legal_actions:
                print(f"Column {action} is not available.")
                continue
            return action


def _load_checkpoint_agent(agent_type: str, checkpoint_path: str):
    module_name, class_name = {
        "dqn": ("connect4.dqn", "DQNPolicyAgent"),
        "ppo": ("connect4.ppo", "PPOPolicyAgent"),
        "final-boss": ("connect4.strong_dqn", "StrongDQNPolicyAgent"),
    }[agent_type]
    module = import_module(module_name)
    agent_class = getattr(module, class_name)
    return agent_class.from_checkpoint(checkpoint_path)


def _resolve_config(*players: Agent | HumanAgent, fallback: Connect4Config) -> Connect4Config:
    for player in players:
        config = getattr(player, "config", None)
        if isinstance(config, Connect4Config):
            return config
    return fallback


def make_player(player_type: str, seed: int | None, checkpoint_path: str | None) -> Agent | HumanAgent:
    if player_type == "human":
        return HumanAgent()
    if player_type == "final-boss":
        return build_agent(player_type, seed=seed)
    if player_type == "dqn":
        if checkpoint_path is None:
            raise ValueError("A checkpoint path is required when player type is dqn")
        return _load_checkpoint_agent(player_type, checkpoint_path)
    if player_type == "ppo":
        if checkpoint_path is None:
            raise ValueError("A checkpoint path is required when player type is ppo")
        return _load_checkpoint_agent(player_type, checkpoint_path)
    return build_agent(player_type, seed=seed)


def run_game(player_one: Agent | HumanAgent, player_two: Agent | HumanAgent, config: Connect4Config) -> None:
    env = Connect4Env(config)
    env.reset(start_player=1)

    print(env.render())
    print()

    players = {1: player_one, -1: player_two}
    while not env.done:
        acting_player = env.current_player
        marker = "X" if acting_player == 1 else "O"
        player = players[acting_player]
        print(f"Player {marker} ({player.name}) to move.")
        action = player.select_action(env)
        _, reward, done, info = env.step(action)
        print(f"Played column {action}")
        print(env.render())
        print()

        if done:
            winner = info["winner"]
            if winner == 0:
                print("Game ended in a draw.")
            else:
                winner_marker = "X" if winner == 1 else "O"
                print(f"Player {winner_marker} wins with reward {reward:.1f}.")
            return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play a configurable Connect-style game in the terminal.")
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--cols", type=int, default=7)
    parser.add_argument("--connect-n", type=int, default=4)
    parser.add_argument("--player1", choices=["human", "random", "heuristic", "dqn", "ppo", "final-boss"], default="human")
    parser.add_argument("--player2", choices=["human", "random", "heuristic", "dqn", "ppo", "final-boss"], default="heuristic")
    parser.add_argument("--player1-checkpoint")
    parser.add_argument("--player2-checkpoint")
    parser.add_argument("--agent-type", choices=["dqn", "ppo", "final-boss"])
    parser.add_argument("--agent-checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.agent_checkpoint is not None:
            if args.agent_type is None:
                raise ValueError("--agent-type is required when using --agent-checkpoint")
            player_one = make_player(args.agent_type, seed=args.seed, checkpoint_path=args.agent_checkpoint)
            player_two = HumanAgent()
            fallback_config = Connect4Config(rows=args.rows, cols=args.cols, connect_n=args.connect_n)
            config = _resolve_config(player_one, fallback=fallback_config)
        else:
            player_one = make_player(args.player1, seed=args.seed, checkpoint_path=args.player1_checkpoint)
            player_two = make_player(args.player2, seed=args.seed + 1, checkpoint_path=args.player2_checkpoint)
            fallback_config = Connect4Config(rows=args.rows, cols=args.cols, connect_n=args.connect_n)
            config = _resolve_config(player_one, player_two, fallback=fallback_config)

        run_game(player_one, player_two, config)
    except KeyboardInterrupt:
        print("\nGame aborted.")
    except ValueError as exc:
        print(f"Error: {exc}")


if __name__ == "__main__":
    main()