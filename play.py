"""
Interactive game vs the rule-based, CFR, or hemobot opponent.

Usage:
    python3 play.py                       # rule-based opponent
    python3 play.py --bot hemo            # hemobot opponent
    python3 play.py --cfr cfr_bot.pkl     # trained CFR opponent
    python3 play.py --bot hemo --openplay # see bot's hole cards live
"""

import argparse
import pickle
from poker_env import (
    new_game, apply_action, info_set_key, legal_actions,
    FOLD, CALL, RAISE
)
from rule_bot import decide_action as rule_decide
from cfr_bot import CFRBot
from hemo_bot import decide_action as hemo_decide
from hand_evaluator import evaluate_hand_strength, describe_hand

BIG_BLIND = 2.0
ACTION_LABELS = {FOLD: "fold", CALL: "call/check", RAISE: "bet/raise"}


# ── Bet sizing ────────────────────────────────────────────────────────────────

def _bot_bet_size(state, player: int) -> float:
    """Compute a reasonable bet size for the bot."""
    stack = state.stacks[player]
    if state.street == "preflop":
        size = max(state.current_bet * 3, BIG_BLIND * 3)
    else:
        size = max(state.pot * 0.67, BIG_BLIND)
    return min(size, stack)


def _prompt_raise_size(state, player: int) -> float:
    """Ask the human how much to raise. Clamped to their stack."""
    stack = state.stacks[player]
    already_in = state.street_invested.get(player, 0.0)
    call_amount = max(0.0, state.current_bet - already_in)
    max_raise = stack - call_amount          # extra chips available after calling
    max_raise = max(max_raise, 0.0)
    min_raise = min(max(state.last_raise_size, state.big_blind), max_raise)

    if state.street == "preflop":
        default = max(state.current_bet * 3 - already_in, BIG_BLIND * 3)
    else:
        default = state.pot * 0.67

    default = max(min(default, max_raise), min_raise)

    prompt = (f"  Raise by how much? [default={default:.0f}, "
              f"min={min_raise:.0f}, max={max_raise:.0f}, 'a'=all-in]: ")
    while True:
        raw = input(prompt).strip()
        if raw == "":
            return default
        if raw.lower() == "a":
            return max_raise
        try:
            amount = float(raw)
            if amount <= 0:
                print("  Must be positive.")
                continue
            if amount > max_raise:
                print(f"  Only {max_raise:.1f} available to raise — going all-in.")
                return max_raise
            if amount < min_raise:
                print(f"  Minimum raise is {min_raise:.1f} — raising that instead.")
                return min_raise
            return amount
        except ValueError:
            print("  Enter a number, press Enter for default, or 'a' for all-in.")


# ── Display ───────────────────────────────────────────────────────────────────

def _print_hud(state, human_player: int, bot_player: int, openplay: bool) -> None:
    hole = state.hole_cards[human_player]
    bot_hole = state.hole_cards[bot_player]
    community = state.community

    print()
    print(f"  Street          : {state.street.upper()}")
    print(f"  Community       : {' '.join(community) if community else '(none)'}")
    print(f"  Your cards      : {' '.join(hole)}", end="")
    if community:
        strength = evaluate_hand_strength(hole, community)
        hand_desc = describe_hand(hole, community)
        print(f"  →  {hand_desc} ({strength:.2f})", end="")
    print()
    if openplay:
        bot_desc = ""
        if community:
            bot_strength = evaluate_hand_strength(bot_hole, community)
            bot_hand_desc = describe_hand(bot_hole, community)
            bot_desc = f"  →  {bot_hand_desc} ({bot_strength:.2f})"
        print(f"  Bot cards       : {' '.join(bot_hole)}{bot_desc}")
    print(f"  Pot             : {state.pot:.1f}")
    print(f"  Your stack      : {state.stacks[human_player]:.1f}  "
          f"(in this street: {state.street_invested.get(human_player, 0):.1f})")
    print(f"  Bot stack       : {state.stacks[bot_player]:.1f}  "
          f"(in this street: {state.street_invested.get(bot_player, 0):.1f})")
    to_call = max(0.0, state.current_bet - state.street_invested.get(human_player, 0.0))
    if to_call > 0:
        print(f"  To call         : {to_call:.1f}")


def _prompt_action(state, human_player: int, legal: list[int]) -> tuple[int, float]:
    """Return (action, bet_size). bet_size is only used for RAISE."""
    to_call = max(0.0, state.current_bet - state.street_invested.get(human_player, 0.0))
    names = {FOLD: "fold", CALL: "call" if to_call > 0 else "check",
             RAISE: "raise" if to_call > 0 else "bet"}
    prompt = "  Action [" + ", ".join(f"{a}={names[a]}" for a in legal) + "]: "

    while True:
        choice = input(prompt).strip()
        if choice.isdigit() and int(choice) in legal:
            action = int(choice)
            if action == RAISE:
                return RAISE, _prompt_raise_size(state, human_player)
            return action, 0.0
        print("  Invalid — enter " + " or ".join(str(a) for a in legal) + ".")


# ── Hand loop ─────────────────────────────────────────────────────────────────

def play_hand(bot, human_player: int = 0, openplay: bool = False) -> float:
    state = new_game(num_players=2)
    bot_player = 1 - human_player
    pos = state.positions.get(human_player, "?")

    print("\n" + "=" * 55)
    print(f"  New hand  |  You: Player {human_player} ({pos})  |  "
          f"Buy-in: {state.stacks[human_player] + state.street_invested.get(human_player, 0):.0f}")

    while not state.terminal:
        p = state.current_player

        if p == human_player:
            _print_hud(state, human_player, bot_player, openplay)
            action, bet_size = _prompt_action(state, human_player, legal_actions(state))
            apply_action(state, action, bet_size)
        else:
            action = _bot_decide(state, bot, p)
            bet_size = _bot_bet_size(state, p) if action == RAISE else 0.0
            before = state.total_invested_all.get(p, 0.0)
            apply_action(state, action, bet_size)
            label = ACTION_LABELS[action]
            if action == RAISE:
                # Read the real amount back off the state — the env may have
                # raised it to the minimum or clamped it to the bot's stack.
                committed = state.total_invested_all.get(p, before) - before
                print(f"\n  Bot action      : {label} {committed:.0f}")
            else:
                print(f"\n  Bot action      : {label}")

    # Showdown / hand over
    print("\n" + "-" * 55)
    if openplay or state.went_to_showdown:
        bot_hole = state.hole_cards[bot_player]
        print(f"  Bot cards       : {' '.join(bot_hole)}")
        if state.community:
            print(f"  Community       : {' '.join(state.community)}")
    payoff = state.returns.get(human_player, 0.0)
    result = "WON" if payoff > 0 else ("LOST" if payoff < 0 else "CHOPPED")
    print(f"  Result          : {result}  {payoff:+.1f}")
    return payoff


def _bot_decide(state, bot, player: int) -> int:
    if isinstance(bot, CFRBot):
        return bot.get_action(info_set_key(state, player), legal_actions(state))
    elif bot == "hemo":
        return hemo_decide(state, player, legal_actions(state))
    else:
        hole = state.hole_cards[player]
        return rule_decide(hole, state.community, legal_actions(state), state.pot, state.stacks[player])


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bot", type=str, default="rule", choices=["rule", "hemo"],
                        help="Opponent bot type")
    parser.add_argument("--cfr", type=str, default=None,
                        help="Path to trained CFR bot .pkl file")
    parser.add_argument("--hands", type=int, default=10,
                        help="Number of hands to play")
    parser.add_argument("--openplay", action="store_true",
                        help="Show bot's hole cards throughout the hand")
    args = parser.parse_args()

    if args.cfr:
        bot = CFRBot()
        with open(args.cfr, "rb") as f:
            data = pickle.load(f)
        bot.regret_sum.update(data["regret_sum"])
        bot.strategy_sum.update(data["strategy_sum"])
        print(f"Loaded CFR bot from {args.cfr}")
    elif args.bot == "hemo":
        bot = "hemo"
        print("Opponent: hemobot")
    else:
        bot = None
        print("Opponent: rule bot")

    if args.openplay:
        print("Open-play mode ON — bot's cards are visible")

    total = 0.0
    for i in range(args.hands):
        total += play_hand(bot, openplay=args.openplay)

    print("\n" + "=" * 55)
    print(f"  Total over {args.hands} hands: {total:+.1f}")


if __name__ == "__main__":
    main()
