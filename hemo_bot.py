"""
hemobot — position-aware rule-based bot.

Opening ranges are keyed by POSITION NAME, not table size.
Table size determines which position names get assigned to seats (in poker_env.py).
The ranges below cover all positions from BTN (loosest) to UTG (tightest).

6-max anchors (user-specified):
  BTN 48% | CO 29% | HJ 22% | SB 35%
EP ranges for 7–9 max are interpolated from those anchors toward standard GTO theory.

Preflop branching keys off how many RAISES have gone in, not on whether the
player owes chips: the SB and the heads-up button owe part of a blind in an
unopened pot, and they should be opening there, not defending.

BB defense is keyed by the OPENER's position, not the BB's own, because how
wide the BB continues depends on how wide the raiser is. It is much wider than
any opening range — the BB is already a blind in and closes the action.

Preflop aggression responses:
  - 4bet: AA, KK, QQ, AK  (AA/KK willing to jam; QQ/AK 4bet but not jam)
  - Call in position: all pocket pairs, all suited aces, AQo+, KQs, KJs+
  - Call out of position: QQ+, AK only
  - Everything else: fold

Post-flop (unchanged):
  - Facing bet + strong hand (>0.85): raise
  - Facing bet otherwise: fold
  - Strong hand (>0.75): bet for value
  - In position as preflop aggressor: c-bet
  - Otherwise: check
"""

from poker_env import GameState, FOLD, CALL, RAISE
from hand_evaluator import evaluate_hand_strength

RANKS = "23456789TJQKA"  # index 0 = 2, index 12 = A


def _hi_lo(hole: list[str]) -> tuple[int, int, bool]:
    """Return (hi_rank_idx, lo_rank_idx, is_suited)."""
    r0 = RANKS.index(hole[0][0])
    r1 = RANKS.index(hole[1][0])
    return max(r0, r1), min(r0, r1), hole[0][1] == hole[1][1]


# ── Opening range functions ────────────────────────────────────────────────────
# Each function returns True if (hi, lo, suited, pair) is in that position's range.
# Rank indices: 2=0 3=1 4=2 5=3 6=4 7=5 8=6 9=7 T=8 J=9 Q=10 K=11 A=12

def _range_btn(hi, lo, suited, pair) -> bool:
    """BTN ~48%"""
    if pair: return True
    gap = hi - lo
    if suited:
        if hi == 12: return True                  # A2s+
        if hi == 11: return True                  # K2s+
        if hi == 10 and lo >= 3: return True      # Q5s+
        if hi ==  9 and lo >= 5: return True      # J7s+
        if hi ==  8 and lo >= 5: return True      # T7s+
        if gap == 2 and hi >= 5: return True      # 75s+, 86s+, 97s+
        if hi ==  4 and lo == 3: return True      # 65s
        if hi ==  3 and lo == 2: return True      # 54s
    else:
        if hi == 12: return True                  # A2o+
        if hi == 11 and lo >= 6: return True      # K8o+
        if hi == 10 and lo >= 7: return True      # Q9o+
        if hi ==  9 and lo >= 7: return True      # J9o+
        if hi ==  8 and lo == 7: return True      # T9o
    return False


def _range_sb(hi, lo, suited, pair) -> bool:
    """SB ~35% — pure raise, no limp"""
    if pair: return True
    gap = hi - lo
    if suited:
        if hi == 12: return True                  # A2s+
        if hi == 11 and lo >= 3: return True      # K5s+
        if hi == 10 and lo >= 6: return True      # Q8s+
        if hi ==  9 and lo >= 6: return True      # J8s+
        if gap == 2 and hi >= 6: return True      # 86s+, 97s+
        if hi ==  5 and lo == 4: return True      # 76s
        if hi ==  4 and lo == 3: return True      # 65s
    else:
        if hi == 12: return True                  # A2o+
        if hi == 11 and lo >= 6: return True      # K8o+
        if hi == 10 and lo >= 7: return True      # Q9o+
        if hi ==  9 and lo >= 7: return True      # J9o+
    return False


def _range_co(hi, lo, suited, pair) -> bool:
    """CO ~29%"""
    if pair: return True
    gap = hi - lo
    if suited:
        if hi == 12: return True                  # A2s+
        if hi == 11 and lo >= 7: return True      # K9s+
        if hi == 10 and lo >= 7: return True      # Q9s+
        if hi ==  9 and lo >= 7: return True      # J9s+
        if gap == 1 and hi >= 4: return True      # 65s+ (all suited connectors)
        if hi ==  3 and lo == 2: return True      # 54s
        if gap == 2 and hi >= 8: return True      # T8s+
    else:
        if hi == 12 and lo >= 9: return True      # AJo+
        if hi == 11 and lo == 10: return True     # KQo
        if hi == 10 and lo ==  9: return True     # QJo
        if hi ==  9 and lo ==  8: return True     # JTo
    return False


def _range_hj(hi, lo, suited, pair) -> bool:
    """HJ ~22%"""
    if pair: return hi >= 3                       # 55+
    if suited:
        if hi == 12: return True                  # A2s+ (all suited aces, incl. AJs+)
        if hi == 11 and lo >= 8: return True      # KTs+
        if hi == 10 and lo >= 8: return True      # QTs+
        if hi ==  9 and lo == 8: return True      # JTs
    else:
        if hi == 12 and lo >= 9: return True      # AJo+
        if hi == 11 and lo == 10: return True     # KQo
    return False


def _range_lj(hi, lo, suited, pair) -> bool:
    """LJ ~17% — tighten HJ: drop weaker suited aces and AJo"""
    if pair: return hi >= 3                       # 55+
    if suited:
        if hi == 12 and lo >= 3: return True      # A5s+ (drop A2s–A4s)
        if hi == 11 and lo >= 8: return True      # KTs+
        if hi == 10 and lo >= 8: return True      # QTs+
        if hi ==  9 and lo == 8: return True      # JTs
    else:
        if hi == 12 and lo >= 10: return True     # AQo+ (drop AJo)
        if hi == 11 and lo == 10: return True     # KQo
    return False


def _range_utg(hi, lo, suited, pair) -> bool:
    """UTG (6-max) ~14%"""
    if pair: return hi >= 5                       # 77+
    if suited:
        if hi == 12 and lo >= 7: return True      # A9s+
        if hi == 11 and lo >= 8: return True      # KTs+
        if hi == 10 and lo >= 8: return True      # QTs+
        if hi ==  9 and lo == 8: return True      # JTs
    else:
        if hi == 12 and lo >= 10: return True     # AQo+
        if hi == 11 and lo == 10: return True     # KQo
    return False


def _range_utg1(hi, lo, suited, pair) -> bool:
    """UTG1 (8-max) ~12%"""
    if pair: return hi >= 6                       # 88+
    if suited:
        if hi == 12 and lo >= 8: return True      # ATs+
        if hi == 11 and lo >= 10: return True     # KQs
        if hi ==  9 and lo == 8: return True      # JTs
    else:
        if hi == 12 and lo >= 10: return True     # AQo+
    return False


def _range_utg2(hi, lo, suited, pair) -> bool:
    """UTG2 (9-max) ~10%"""
    if pair: return hi >= 7                       # 99+
    if suited:
        if hi == 12 and lo >= 9: return True      # AJs+
        if hi == 11 and lo == 10: return True     # KQs
    else:
        if hi == 12 and lo >= 11: return True     # AKo
    return False


_RANGE_FN = {
    "BTN":  _range_btn,
    "SB":   _range_sb,
    "CO":   _range_co,
    "HJ":   _range_hj,
    "LJ":   _range_lj,
    "UTG":  _range_utg,
    "UTG1": _range_utg1,
    "UTG2": _range_utg2,
}


def _in_opening_range(hole: list[str], position: str) -> bool:
    hi, lo, suited = _hi_lo(hole)
    fn = _RANGE_FN.get(position)
    if fn is None:
        return False  # BB / unknown — no open range
    return fn(hi, lo, suited, hi == lo)


# ── 3bet defense ──────────────────────────────────────────────────────────────

def _should_4bet(hi: int, lo: int, suited: bool, pair: bool) -> bool:
    """AA, KK, QQ, AK — always 4bet. AA/KK can jam; QQ/AK 4bet but not jam (same RAISE action here)."""
    if pair and hi >= 10: return True     # QQ, KK, AA
    if hi == 12 and lo == 11: return True  # AKs / AKo
    return False


def _call_3bet_ip(hi: int, lo: int, suited: bool, pair: bool) -> bool:
    """In position: call with pairs, suited aces, and top of range."""
    if pair: return True                          # all pocket pairs
    if suited and hi == 12: return True           # all suited aces
    if hi == 12 and lo >= 10: return True         # AQo+
    if hi == 11 and lo >= 9 and suited: return True  # KJs+
    if hi == 11 and lo == 10: return True         # KQo
    return False


def _call_3bet_oop(hi: int, lo: int, suited: bool, pair: bool) -> bool:
    """Out of position: very tight — QQ+, AK only (JJ/TT are folds to keep it simple)."""
    if pair and hi >= 10: return True             # QQ+
    if hi == 12 and lo == 11: return True         # AK
    return False


# ── BB defense ────────────────────────────────────────────────────────────────
# The BB is a different problem from opening. It already has a blind in and it
# closes the action, so it is getting a price to continue and defends far wider
# than any opening range — roughly 33% against an early open, up to ~65%
# against the SB.
#
# How wide depends on how wide the opener is, so the tiers below are keyed to
# hemobot's own opening table rather than to seat names: LJ opens ~17% at a
# 6-max table and is the early seat there, while HJ (~22%) and CO (~29%) are
# progressively looser.
#
# Each function returns "3bet", "call" or "fold". The 3bet ranges are polarised
# the way solvers build them — a value core plus the suited wheel aces
# (A2s–A5s), which are the best bluffs because they block AA/AK and flop
# nut-flush and wheel draws. Everything not worth 3betting and not folded just
# calls, since the BB rarely wants to bloat the pot out of position.

def _bb_vs_early(hi, lo, suited, pair) -> str:
    """vs UTG / UTG1 / UTG2 / LJ opens (~10–17%) — defend ~33%."""
    gap = hi - lo
    if pair and hi >= 10: return "3bet"            # QQ+
    if hi == 12 and lo == 11: return "3bet"        # AK
    if suited and hi == 12 and lo == 10: return "3bet"   # AQs
    if suited and hi == 12 and lo <= 3: return "3bet"    # A2s–A5s bluffs
    if pair: return "call"                         # 22–JJ
    if suited:
        if hi == 12: return "call"                 # A2s–AJs
        if hi == 11 and lo >= 5: return "call"     # K7s+
        if hi == 10 and lo >= 6: return "call"     # Q8s+
        if hi ==  9 and lo >= 6: return "call"     # J8s+
        if hi ==  8 and lo >= 5: return "call"     # T7s+
        if gap == 1 and hi >= 3: return "call"     # 54s+
        if gap == 2 and hi >= 5: return "call"     # 75s+
    else:
        if hi == 12 and lo >= 6: return "call"     # A8o+
        if hi == 11 and lo >= 7: return "call"     # K9o+
        if hi == 10 and lo >= 7: return "call"     # Q9o+
        if hi ==  9 and lo >= 7: return "call"     # J9o+
        if hi ==  8 and lo ==  7: return "call"    # T9o
    return "fold"


def _bb_vs_hj(hi, lo, suited, pair) -> str:
    """vs an HJ open (~22%) — defend ~42%."""
    gap = hi - lo
    if pair and hi >= 9: return "3bet"             # JJ+
    if hi == 12 and lo == 11: return "3bet"        # AK
    if suited and hi == 12 and lo == 10: return "3bet"   # AQs
    if suited and hi == 11 and lo == 10: return "3bet"   # KQs
    if suited and hi == 12 and lo <= 3: return "3bet"    # A2s–A5s bluffs
    if pair: return "call"
    if suited:
        if hi == 12: return "call"                 # A6s–AJs
        if hi == 11 and lo >= 3: return "call"     # K5s+
        if hi == 10 and lo >= 4: return "call"     # Q6s+
        if hi ==  9 and lo >= 4: return "call"     # J6s+
        if hi ==  8 and lo >= 4: return "call"     # T6s+
        if gap == 1 and hi >= 2: return "call"     # 43s+
        if gap == 2 and hi >= 3: return "call"     # 53s+
    else:
        if hi == 12 and lo >= 4: return "call"     # A6o+
        if hi == 11 and lo >= 6: return "call"     # K8o+
        if hi == 10 and lo >= 6: return "call"     # Q8o+
        if hi ==  9 and lo >= 6: return "call"     # J8o+
        if hi ==  8 and lo >= 6: return "call"     # T8o+
        if hi ==  7 and lo ==  6: return "call"    # 98o
    return "fold"


def _bb_vs_co(hi, lo, suited, pair) -> str:
    """vs a CO open (~29%) — defend ~48%."""
    gap = hi - lo
    if pair and hi >= 9: return "3bet"             # JJ+
    if hi == 12 and lo >= 10: return "3bet"        # AQ+
    if suited and hi == 11 and lo == 10: return "3bet"   # KQs
    if suited and hi == 12 and lo <= 3: return "3bet"    # A2s–A5s bluffs
    if pair: return "call"
    if suited:
        if hi >= 11: return "call"                 # every suited ace and king
        if hi == 10 and lo >= 2: return "call"     # Q4s+
        if hi ==  9 and lo >= 3: return "call"     # J5s+
        if hi ==  8 and lo >= 3: return "call"     # T5s+
        if gap <= 2: return "call"                 # connectors and one-gappers
        if gap == 3 and hi >= 5: return "call"     # 74s+
    else:
        if hi == 12 and lo >= 2: return "call"     # A4o+
        if hi == 11 and lo >= 5: return "call"     # K7o+
        if hi == 10 and lo >= 5: return "call"     # Q7o+
        if hi ==  9 and lo >= 5: return "call"     # J7o+
        if hi ==  8 and lo >= 5: return "call"     # T7o+
        if hi ==  7 and lo >= 5: return "call"     # 97o+
        if hi ==  6 and lo ==  5: return "call"    # 87o
    return "fold"


def _bb_vs_btn(hi, lo, suited, pair) -> str:
    """vs a BTN open (~48%), and the heads-up button — defend ~57%."""
    gap = hi - lo
    if pair and hi >= 8: return "3bet"             # TT+
    if hi == 12 and lo >= 10: return "3bet"        # AQ+
    if suited and hi == 12 and lo == 9: return "3bet"    # AJs
    if suited and hi == 11 and lo == 10: return "3bet"   # KQs
    if suited and hi == 12 and lo <= 3: return "3bet"    # A2s–A5s bluffs
    if suited and hi == 11 and 3 <= lo <= 5: return "3bet"  # K5s–K7s bluffs
    if pair: return "call"                         # 22–99
    if suited:
        if hi >= 10: return "call"                 # every suited ace, king, queen
        if hi ==  9 and lo >= 1: return "call"     # J3s+
        if hi ==  8 and lo >= 1: return "call"     # T3s+
        if gap <= 2: return "call"                 # connectors and one-gappers
        if gap == 3 and hi >= 3: return "call"     # 52s+
    else:
        if hi == 12: return "call"                 # A2o+
        if hi == 11 and lo >= 5: return "call"     # K7o+
        if hi == 10 and lo >= 5: return "call"     # Q7o+
        if hi ==  9 and lo >= 5: return "call"     # J7o+
        if hi ==  8 and lo >= 5: return "call"     # T7o+
        if hi ==  7 and lo >= 4: return "call"     # 96o+
        if hi ==  6 and lo ==  5: return "call"    # 87o
    return "fold"


def _bb_vs_sb(hi, lo, suited, pair) -> str:
    """
    vs an SB open (~35%) — defend ~65%.

    The widest tier, because this is the one spot where the BB has position:
    post-flop the SB acts first and the BB closes every street.
    """
    if pair and hi >= 7: return "3bet"             # 99+
    if hi == 12 and lo >= 10: return "3bet"        # AQ+
    if suited and hi == 12 and lo == 9: return "3bet"    # AJs
    if suited and hi == 11 and lo >= 9: return "3bet"    # KJs+
    if suited and hi == 10 and lo ==  9: return "3bet"   # QJs
    if suited and hi == 12 and lo <= 3: return "3bet"    # A2s–A5s bluffs
    if pair: return "call"
    if suited: return "call"                       # every suited hand has a price
    if hi == 12: return "call"                     # A2o+
    if hi == 11 and lo >= 2: return "call"         # K4o+
    if hi == 10 and lo >= 4: return "call"         # Q6o+
    if hi ==  9 and lo >= 4: return "call"         # J6o+
    if hi ==  8 and lo >= 4: return "call"         # T6o+
    if hi ==  7 and lo >= 4: return "call"         # 96o+
    if hi ==  6 and lo >= 4: return "call"         # 86o+
    if hi ==  5 and lo ==  4: return "call"        # 76o
    return "fold"


_BB_DEFENSE_FN = {
    "SB":   _bb_vs_sb,
    "BTN":  _bb_vs_btn,
    "CO":   _bb_vs_co,
    "HJ":   _bb_vs_hj,
    "LJ":   _bb_vs_early,
    "UTG":  _bb_vs_early,
    "UTG1": _bb_vs_early,
    "UTG2": _bb_vs_early,
}


def _bb_defense(hi, lo, suited, pair, opener_position: str) -> str:
    """Return "3bet", "call" or "fold" for the BB facing a single open."""
    # An unknown opener gets the tightest tier rather than the widest.
    fn = _BB_DEFENSE_FN.get(opener_position, _bb_vs_early)
    return fn(hi, lo, suited, pair)


def _bb_option_raise(hi, lo, suited, pair) -> bool:
    """
    Nobody raised, so the BB is in for free and never folds. Raise the hands
    that want a bigger pot against limpers; check the rest.
    """
    if pair and hi >= 8: return True               # TT+
    if hi == 12 and lo >= 10: return True          # AQ+
    if suited and hi == 12 and lo >= 9: return True  # AJs+
    if suited and hi == 11 and lo == 10: return True # KQs
    return False


# ── Helpers ────────────────────────────────────────────────────────────────────

def _is_in_position(state: GameState, player: int) -> bool:
    return player == state.in_position_player and state.street != "preflop"


# ── Main decision function ─────────────────────────────────────────────────────

def decide_action(state: GameState, player: int, legal: list[int]) -> int:
    hole = state.hole_cards[player]
    position = state.positions.get(player, "UTG")
    hi, lo, suited = _hi_lo(hole)
    pair = hi == lo

    def pick(*preferred):
        for a in preferred:
            if a in legal:
                return a
        return legal[0]

    # ── PREFLOP ──────────────────────────────────────────────────────────────
    if state.street == "preflop":
        # Count raises, not chips owed. The SB — and the heads-up button, which
        # posts the small blind — owes the rest of a blind before anyone has
        # raised; that is an unopened pot, not a raise to defend against.
        raises = state.raises_this_street

        if raises == 0:
            if position == "BB":
                # The option. Checking is free, so the BB never folds here.
                if _bb_option_raise(hi, lo, suited, pair):
                    return pick(RAISE, CALL)
                return pick(CALL)
            if _in_opening_range(hole, position):
                return pick(RAISE, CALL)
            return pick(FOLD)

        # Facing a single open. The BB closes the action and is already a blind
        # in, so it defends on pot odds with its own ranges.
        if raises == 1 and position == "BB":
            opener = state.positions.get(state.preflop_aggressor, "UTG")
            move = _bb_defense(hi, lo, suited, pair, opener)
            if move == "3bet":
                return pick(RAISE, CALL)
            if move == "call":
                return pick(CALL)
            return pick(FOLD)

        if _should_4bet(hi, lo, suited, pair):
            return pick(RAISE, CALL)   # 4bet (AA/KK: happy to jam; QQ/AK: 4bet small)
        ip = player == state.in_position_player
        if ip and _call_3bet_ip(hi, lo, suited, pair):
            return pick(CALL)
        if not ip and _call_3bet_oop(hi, lo, suited, pair):
            return pick(CALL)
        return pick(FOLD)

    # ── POST-FLOP ─────────────────────────────────────────────────────────────
    strength = evaluate_hand_strength(hole, state.community)
    in_pos = _is_in_position(state, player)
    is_aggressor = state.preflop_aggressor == player

    if state.facing_bet:
        if strength > 0.85:
            return pick(RAISE, CALL)   # raise for value
        return pick(FOLD)              # low stakes players don't bluff

    if strength > 0.75:
        return pick(RAISE, CALL)       # bet strong hands for value

    if in_pos and is_aggressor:
        return pick(RAISE, CALL)       # c-bet in position

    return pick(CALL)                  # check
