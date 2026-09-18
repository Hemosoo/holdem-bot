"""
CFR self-play training loop.

Usage:
    python train.py --iterations 10000
"""

import argparse
import pickle
from cfr_bot import CFRBot
from poker_env import (
    new_game, apply_action, info_set_key, legal_actions,
    FOLD, CALL, RAISE
)


def traverse(state, bot: CFRBot, player: int) -> float:
    if state.terminal:
        return state.returns.get(player, 0.0)

    current = state.current_player
    info_set = info_set_key(state, current)
    actions = legal_actions(state)
    strategy = bot.get_strategy(info_set)

    action_values = {}
    node_value = 0.0

    for a in actions:
        import copy
        next_state = copy.deepcopy(state)
        apply_action(next_state, a)
        action_values[a] = traverse(next_state, bot, player)
        node_value += strategy[a] * action_values[a]

    if current == player:
        for a in actions:
            regret = action_values[a] - node_value
            bot.update_regrets(info_set, a, regret)

    return node_value


def train(iterations: int = 5000) -> CFRBot:
    bot = CFRBot()
    for i in range(iterations):
        for player in range(2):
            state = new_game(num_players=2)
            traverse(state, bot, player)
        if (i + 1) % 1000 == 0:
            print(f"Iteration {i + 1}/{iterations}")
    return bot


def main():
    parser = argparse.ArgumentParser(description="Train CFR poker bot")
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--output", type=str, default="cfr_bot.pkl")
    args = parser.parse_args()

    print(f"Training for {args.iterations} iterations...")
    bot = train(args.iterations)

    with open(args.output, "wb") as f:
        pickle.dump({"regret_sum": dict(bot.regret_sum), "strategy_sum": dict(bot.strategy_sum)}, f)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
