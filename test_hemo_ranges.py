"""
Frequency and sanity tests for hemobot's BB defense ranges.

Every hand class is weighted by its combo count (pair 6, suited 4, offsuit 12)
so the percentages are real range shares, not counts of grid squares.

    python3 test_hemo_ranges.py
"""

import sys

from hemo_bot import (
    _bb_vs_early, _bb_vs_hj, _bb_vs_co, _bb_vs_btn, _bb_vs_sb, _bb_defense,
)

RANKS = "23456789TJQKA"
TOLERANCE = 0.05

# Typical solver BB defense totals vs a ~2.5x open, 100bb, no rake.
TIERS = [
    ("vs early (UTG-LJ)", _bb_vs_early, 0.33),
    ("vs HJ",             _bb_vs_hj,    0.42),
    ("vs CO",             _bb_vs_co,    0.48),
    ("vs BTN",            _bb_vs_btn,   0.57),
    ("vs SB",             _bb_vs_sb,    0.65),
]

failures = []


def check(name, ok, extra=""):
    print(("  ok   " if ok else "  FAIL ") + name + ("" if ok else f"   {extra}"))
    if not ok:
        failures.append(name)


def share(fn):
    """Return {"3bet": pct, "call": pct, "fold": pct} over all 1326 combos."""
    totals = {"3bet": 0, "call": 0, "fold": 0}
    for hi in range(13):
        for lo in range(hi + 1):
            for suited in ([False] if hi == lo else [True, False]):
                pair = hi == lo
                weight = 6 if pair else (4 if suited else 12)
                totals[fn(hi, lo, suited, pair)] += weight
    combos = sum(totals.values())
    assert combos == 1326, combos
    return {k: v / combos for k, v in totals.items()}


print("Defense frequency vs each opener")
defends = []
for label, fn, target in TIERS:
    pct = share(fn)
    defend = pct["3bet"] + pct["call"]
    defends.append(defend)
    check(f"{label:<20} defends {defend:5.1%} (target {target:.0%})",
          abs(defend - target) <= TOLERANCE)

print("\nRange shape")
check("defense widens as the opener loosens",
      all(a < b for a, b in zip(defends, defends[1:])), defends)
check("3bet frequency stays polarised (4-16%)",
      all(0.04 <= share(fn)["3bet"] <= 0.16 for _, fn, _ in TIERS),
      [f"{share(fn)['3bet']:.1%}" for _, fn, _ in TIERS])

print("\nHands that must behave the same against every opener")
for name, hi, lo, suited, pair, want in [
    ("AA",  12, 12, False, True,  "3bet"),
    ("KK",  11, 11, False, True,  "3bet"),
    ("AKo", 12, 11, False, False, "3bet"),
    ("A5s", 12,  3, True,  False, "3bet"),   # wheel-ace bluff in every tier
    ("72o",  5,  0, False, False, "fold"),   # the worst hand, never defended
    ("32o",  1,  0, False, False, "fold"),
]:
    got = {label: fn(hi, lo, suited, pair) for label, fn, _ in TIERS}
    check(f"{name:<4} is {want} vs every opener",
          all(v == want for v in got.values()), got)

print("\nPot odds mean the BB defends wider than the opener opens")
# hemobot opens BTN ~48%; the BB must defend more than that against it.
check("vs BTN defense (57%) exceeds the BTN opening range (48%)",
      defends[3] > 0.48, defends[3])
check("vs early defense (33%) exceeds the early opening range (17%)",
      defends[0] > 0.17, defends[0])

print("\nDispatch by opener position")
check("LJ routes to the early tier",
      _bb_defense(9, 7, True, False, "LJ") == _bb_vs_early(9, 7, True, False))
check("an unknown opener gets the tightest tier",
      _bb_defense(9, 7, True, False, "???") == _bb_vs_early(9, 7, True, False))
check("every seat name has a tier",
      all(_bb_defense(12, 12, False, True, pos) == "3bet"
          for pos in ("SB", "BTN", "CO", "HJ", "LJ", "UTG", "UTG1", "UTG2")))

print("\n" + ("ALL PASS" if not failures else f"{len(failures)} FAILURES: {failures}"))
sys.exit(1 if failures else 0)
