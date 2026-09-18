"""
Lightweight Texas Hold'em simulation environment. Supports 2–9 players.

Betting rounds are played out properly: every player who can act gets to act,
a check passes the action instead of ending the street, a raise reopens the
action for everyone behind it, and the street only advances once every live
player has acted and matched the current bet.

Seats are laid out in table order, so `(seat + 1) % n` is always the next seat:

    seat: 0    1   2   3     4  …  n-1
          BTN  SB  BB  UTG   …      CO

Heads-up is the usual special case: seat 0 is the BTN and posts the small
blind, seat 1 is the BB. Preflop the action starts left of the BB; post-flop it
starts left of the button, so the BTN closes every post-flop street.

Simplifications kept deliberate:
  - Raises are capped at MAX_RAISES_PER_STREET, which bounds the game tree.
  - An undersized all-in raise does not reopen the action in real poker; here
    players behind it may re-raise rather than only call or fold.
"""

import random
from dataclasses import dataclass, field

RANKS = "23456789TJQKA"
SUITS = "hdsc"
DECK = [r + s for r in RANKS for s in SUITS]

# Action indices
FOLD = 0
CALL = 1
RAISE = 2

MAX_RAISES_PER_STREET = 4  # betting cap — keeps the game tree finite

_EPS = 1e-9


@dataclass
class GameState:
    deck: list = field(default_factory=list)
    hole_cards: dict = field(default_factory=dict)   # player_id -> [card, card]
    community: list = field(default_factory=list)
    pot: float = 0.0
    stacks: dict = field(default_factory=dict)       # player_id -> chips remaining
    current_player: int = 0
    street: str = "preflop"
    terminal: bool = False
    returns: dict = field(default_factory=dict)      # player_id -> net payoff

    # --- bet tracking ---
    current_bet: float = 0.0                         # highest total bet this street
    street_invested: dict = field(default_factory=dict)    # chips put in THIS street per player
    total_invested_all: dict = field(default_factory=dict) # cumulative all streets per player
    big_blind: float = 2.0

    # --- betting round state ---
    folded: set = field(default_factory=set)
    all_in: set = field(default_factory=set)
    acted_this_street: set = field(default_factory=set)
    last_raise_size: float = 0.0                     # size of the last raise increment
    raises_this_street: int = 0

    # --- position tracking ---
    preflop_aggressor: int | None = None
    facing_bet: bool = False                         # does current_player owe chips?
    in_position_player: int = 0                      # last live player to act post-flop
    positions: dict = field(default_factory=dict)    # player_id -> "BTN"/"CO"/etc.

    went_to_showdown: bool = False


# ── Position assignment ────────────────────────────────────────────────────────

# EP position names from tightest (UTG) to loosest (CO).
# Sliced based on how many EP seats exist at a given table size.
_EP_NAMES = ["UTG", "UTG1", "UTG2", "LJ", "HJ", "CO"]

POSITION_TIER: dict[str, int] = {
    "BTN": 1, "SB": 2, "CO": 3, "HJ": 4,
    "LJ": 5, "UTG2": 6, "UTG1": 7, "UTG": 8, "BB": 9,
}


def _assign_positions(num_players: int) -> dict[int, str]:
    """
    Seat 0 = BTN, Seat 1 = SB (BB in HU), Seat 2 = BB.
    Seats 3+ are EP from tightest (UTG) to loosest (CO).
    """
    if num_players == 2:
        return {0: "BTN", 1: "BB"}
    ep_count = num_players - 3
    ep_names = _EP_NAMES[6 - ep_count: 6]
    positions = {0: "BTN", 1: "SB", 2: "BB"}
    for i, name in enumerate(ep_names):
        positions[3 + i] = name
    return positions


def _blind_seats(num_players: int) -> tuple[int, int]:
    """Return (small_blind_seat, big_blind_seat)."""
    return (0, 1) if num_players == 2 else (1, 2)


# ── Turn order ─────────────────────────────────────────────────────────────────

def contenders(state: GameState) -> list[int]:
    """Players who have not folded — they still have a claim on the pot."""
    return [p for p in sorted(state.stacks) if p not in state.folded]


def _can_act(state: GameState, player: int) -> bool:
    return player not in state.folded and player not in state.all_in


def _first_to_act(state: GameState) -> int | None:
    """
    Preflop the action opens left of the big blind; post-flop it opens left of
    the button. Both are a scan forward from a fixed seat.
    """
    n = len(state.stacks)
    if state.street == "preflop":
        start = (_blind_seats(n)[1] + 1) % n
    else:
        start = 1 % n
    for i in range(n):
        p = (start + i) % n
        if _can_act(state, p):
            return p
    return None


def _next_to_act(state: GameState, after: int) -> int | None:
    """Next seat that can act, wrapping around. May return `after` itself if
    they are the only player left able to act."""
    n = len(state.stacks)
    for i in range(1, n + 1):
        p = (after + i) % n
        if _can_act(state, p):
            return p
    return None


def _update_in_position(state: GameState) -> None:
    """The in-position player is whoever acts last post-flop — the button, or
    the nearest live seat before it."""
    n = len(state.stacks)
    order = [(1 + i) % n for i in range(n)]
    live = [p for p in order if p not in state.folded]
    if live:
        state.in_position_player = live[-1]


def _set_current_player(state: GameState, player: int) -> None:
    state.current_player = player
    owed = state.current_bet - state.street_invested.get(player, 0.0)
    state.facing_bet = owed > _EPS


# ── Game setup ─────────────────────────────────────────────────────────────────

def new_game(num_players: int = 2, starting_stack: float = 200.0, big_blind: float = 2.0) -> GameState:
    if not 2 <= num_players <= 9:
        raise ValueError("num_players must be between 2 and 9")

    deck = DECK[:]
    random.shuffle(deck)

    state = GameState(
        deck=deck,
        hole_cards={p: [deck.pop(), deck.pop()] for p in range(num_players)},
        community=[],
        pot=0.0,
        stacks={p: starting_stack for p in range(num_players)},
        street="preflop",
        street_invested={p: 0.0 for p in range(num_players)},
        total_invested_all={p: 0.0 for p in range(num_players)},
        big_blind=big_blind,
        positions=_assign_positions(num_players),
    )

    sb_seat, bb_seat = _blind_seats(num_players)
    _commit(state, sb_seat, big_blind / 2)
    _commit(state, bb_seat, big_blind)

    state.current_bet = state.street_invested[bb_seat]
    state.last_raise_size = big_blind
    # Posting a blind is not acting — the BB still has its option.
    state.acted_this_street = set()

    _update_in_position(state)
    _set_current_player(state, _first_to_act(state))
    return state


# ── Chip movement ──────────────────────────────────────────────────────────────

def _commit(state: GameState, player: int, amount: float) -> None:
    """Move `amount` from a player's stack into the pot, clamped to the stack."""
    amount = min(max(amount, 0.0), state.stacks[player])
    state.stacks[player] -= amount
    state.pot += amount
    state.street_invested[player] = state.street_invested.get(player, 0.0) + amount
    state.total_invested_all[player] = state.total_invested_all.get(player, 0.0) + amount
    if state.stacks[player] <= _EPS:
        state.all_in.add(player)


# ── Action ────────────────────────────────────────────────────────────────────

def apply_action(state: GameState, action: int, bet_size: float = 5.0) -> None:
    """
    bet_size: for RAISE, the amount to put in ON TOP of any call. Raised to the
              minimum legal raise if it is too small, and clamped to the stack.
    """
    if state.terminal:
        return

    p = state.current_player

    if action == FOLD:
        state.folded.add(p)
    elif action == RAISE:
        _apply_raise(state, p, bet_size)
    else:
        _apply_call(state, p)

    state.acted_this_street.add(p)
    _advance(state)


def _apply_call(state: GameState, player: int) -> None:
    owed = max(0.0, state.current_bet - state.street_invested.get(player, 0.0))
    _commit(state, player, owed)


def _apply_raise(state: GameState, player: int, bet_size: float) -> None:
    owed = max(0.0, state.current_bet - state.street_invested.get(player, 0.0))
    min_increment = max(state.last_raise_size, state.big_blind)
    increment = max(bet_size, min_increment)

    _commit(state, player, owed + increment)

    level = state.street_invested[player]
    if level <= state.current_bet + _EPS:
        return  # stack too short to actually raise — it played as a call

    if state.street == "preflop":
        state.preflop_aggressor = player

    actual_increment = level - state.current_bet
    state.current_bet = level
    state.raises_this_street += 1

    if actual_increment >= min_increment - _EPS:
        # A full raise reopens the action for everyone behind it.
        state.last_raise_size = actual_increment
        state.acted_this_street.clear()


def _round_complete(state: GameState) -> bool:
    live = contenders(state)
    if len(live) <= 1:
        return True
    for p in live:
        if p in state.all_in:
            continue
        if p not in state.acted_this_street:
            return False
        if state.street_invested.get(p, 0.0) < state.current_bet - _EPS:
            return False
    return True


def _advance(state: GameState) -> None:
    """Hand the action to the next player, or move the hand forward."""
    while True:
        _update_in_position(state)
        live = contenders(state)

        if len(live) == 1:
            _award_uncontested(state, live[0])
            return

        if not _round_complete(state):
            nxt = _next_to_act(state, state.current_player)
            if nxt is not None:
                _set_current_player(state, nxt)
                return
            # Nobody can act but the round looks open — treat it as closed.

        if state.street == "river":
            _showdown(state)
            return

        _deal_next_street(state)

        # With fewer than two players able to bet, the rest of the board just
        # runs out — keep dealing until the river.
        if sum(1 for p in live if p not in state.all_in) < 2:
            continue

        _set_current_player(state, _first_to_act(state))
        return


# ── Street dealing ─────────────────────────────────────────────────────────────

def _deal_next_street(state: GameState) -> None:
    if state.street == "preflop":
        state.community = [state.deck.pop() for _ in range(3)]
        state.street = "flop"
    elif state.street == "flop":
        state.community.append(state.deck.pop())
        state.street = "turn"
    elif state.street == "turn":
        state.community.append(state.deck.pop())
        state.street = "river"

    state.current_bet = 0.0
    state.street_invested = {p: 0.0 for p in state.stacks}
    state.acted_this_street = set()
    state.last_raise_size = 0.0
    state.raises_this_street = 0
    state.facing_bet = False


# ── Payouts ───────────────────────────────────────────────────────────────────

def _award_uncontested(state: GameState, winner: int) -> None:
    """Everyone else folded — the last player standing takes the pot."""
    state.terminal = True
    state.went_to_showdown = False
    state.stacks[winner] += state.pot
    for p in state.stacks:
        invested = state.total_invested_all.get(p, 0.0)
        state.returns[p] = state.pot - invested if p == winner else -invested


def _showdown(state: GameState) -> None:
    """Evaluate the live hands and split the pot, side pots included."""
    from treys import Card, Evaluator
    evaluator = Evaluator()
    board = [Card.new(c) for c in state.community]

    scores = {                                        # lower = better
        p: evaluator.evaluate(board, [Card.new(c) for c in state.hole_cards[p]])
        for p in contenders(state)
    }

    state.terminal = True
    state.went_to_showdown = True
    _distribute(state, scores)


def _distribute(state: GameState, scores: dict[int, int]) -> None:
    """
    Split the pot into layers at each all-in level. Each layer is contested
    only by players who put in at least that much, so short stacks can win the
    part of the pot they paid for and no more.
    """
    contributions = {p: state.total_invested_all.get(p, 0.0) for p in state.stacks}
    winnings = {p: 0.0 for p in state.stacks}

    prev = 0.0
    for level in sorted({c for c in contributions.values() if c > 0}):
        layer = {p: min(max(c - prev, 0.0), level - prev) for p, c in contributions.items()}
        amount = sum(layer.values())
        prev = level
        if amount <= _EPS:
            continue

        eligible = [p for p in scores if contributions[p] >= level - _EPS]
        if eligible:
            best = min(scores[p] for p in eligible)
            winners = [p for p in eligible if scores[p] == best]
            for w in winners:
                winnings[w] += amount / len(winners)
        else:
            # Nobody live paid this much — it is an uncalled bet, give it back.
            for p, paid in layer.items():
                winnings[p] += paid

    for p in state.stacks:
        state.stacks[p] += winnings[p]
        state.returns[p] = winnings[p] - contributions[p]


# ── Misc ──────────────────────────────────────────────────────────────────────

def info_set_key(state: GameState, player: int) -> str:
    hole = tuple(sorted(state.hole_cards[player]))
    community = tuple(state.community)
    owed = max(0.0, state.current_bet - state.street_invested.get(player, 0.0))
    facing = "bet" if owed > _EPS else "check"
    return f"{hole}|{community}|{state.street}|r{state.raises_this_street}|{facing}"


def legal_actions(state: GameState) -> list[int]:
    """
    Folding is only offered when there is something to call — folding a free
    check is strictly dominated and only bloats the CFR tree. Raising needs
    chips beyond the call, and stops at the per-street cap.
    """
    if state.terminal:
        return []

    p = state.current_player
    owed = max(0.0, state.current_bet - state.street_invested.get(p, 0.0))

    actions = []
    if owed > _EPS:
        actions.append(FOLD)
    actions.append(CALL)
    if state.stacks[p] > owed + _EPS and state.raises_this_street < MAX_RAISES_PER_STREET:
        actions.append(RAISE)
    return actions
