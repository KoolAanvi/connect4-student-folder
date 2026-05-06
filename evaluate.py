from __future__ import annotations

from typing import Any

from .env import Connect4Config, Connect4Env
from .opponents import Agent


def play_game(
    player_one: Agent,
    player_two: Agent,
    config: Connect4Config | None = None,
    start_player: int = 1,
    render: bool = False,
) -> dict[str, Any]:
    env = Connect4Env(config)
    env.reset(start_player=start_player)

    if render:
        print(env.render())
        print()

    agents = {1: player_one, -1: player_two}
    while not env.done:
        acting_player = env.current_player
        agent = agents[acting_player]
        action = agent.select_action(env)
        _, reward, done, info = env.step(action)

        if render:
            print(f"Player {acting_player:+d} ({agent.name}) played column {action}")
            print(env.render())
            print()

        if done:
            return {
                "winner": info["winner"],
                "reward": reward,
                "move_count": info["move_count"],
                "invalid_action": info["invalid_action"],
                "final_board": env.raw_board(),
            }

    raise RuntimeError("Game loop exited unexpectedly")


def evaluate_agent_pair(
    player_one: Agent,
    player_two: Agent,
    games: int = 100,
    config: Connect4Config | None = None,
    alternate_start: bool = True,
) -> dict[str, float | int]:
    if games <= 0:
        raise ValueError("games must be positive")

    results = {
        "player_one_wins": 0,
        "player_two_wins": 0,
        "draws": 0,
    }

    for game_idx in range(games):
        start_player = 1
        if alternate_start and game_idx % 2 == 1:
            start_player = -1
        outcome = play_game(
            player_one=player_one,
            player_two=player_two,
            config=config,
            start_player=start_player,
            render=False,
        )
        winner = outcome["winner"]
        if winner == 1:
            results["player_one_wins"] += 1
        elif winner == -1:
            results["player_two_wins"] += 1
        else:
            results["draws"] += 1

    total_games = float(games)
    return {
        **results,
        "games": games,
        "player_one_win_rate": results["player_one_wins"] / total_games,
        "player_two_win_rate": results["player_two_wins"] / total_games,
        "draw_rate": results["draws"] / total_games,
    }