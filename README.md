# Poker Bot — Research / Learning Project

A purely theoretical Texas Hold'em bot for exploring AI, game theory, and CFR.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
brew install tesseract   # macOS — required for screen_reader.py OCR
```

---

## Files

| File | Purpose |
|------|---------|
| `poker_env.py` | Texas Hold'em simulator, 2–9 players |
| `hand_evaluator.py` | Hand strength scoring via `treys` (0.0 = worst, 1.0 = best) |
| `rule_bot.py` | Simple baseline bot (folds/calls/raises by hand strength threshold) |
| `cfr_bot.py` | Counterfactual Regret Minimization bot |
| `hemo_bot.py` | Position-aware strategy bot (see below) |
| `test_poker_env.py` | Betting-round, payout and side-pot tests for the simulator |
| `test_hemo_ranges.py` | Frequency and sanity tests for the BB defense ranges |
| `train.py` | Self-play CFR training loop |
| `play.py` | Interactive human-vs-bot game |
| `screen_reader.py` | OpenCV screen reader — detects cards, pot, and stacks from a live poker client |
| `screen_config.json` | Auto-generated region config created by `--calibrate` |

---

## Usage

### Play against hemobot
```bash
python3 play.py --bot hemo
```

### Play against hemobot with bot cards visible
```bash
python3 play.py --bot hemo --openplay
```

### Play against the rule-based bot (default)
```bash
python3 play.py
```

### Train the CFR bot, then play against it
```bash
python3 train.py --iterations 10000 --output cfr_bot.pkl
python3 play.py --cfr cfr_bot.pkl
```

**In-game actions:** `0` = fold, `1` = call/check, `2` = raise/bet

Only legal actions are offered. Fold is hidden when you have nothing to call — checking is free, so folding there is never right. Raise is hidden once you are out of chips or the street has hit its raise cap.

When you raise, you'll be prompted for an amount:
```
Raise by how much? [default=6, max=198, 'a'=all-in]:
```
Press Enter for the suggested size, type any number, or `a` to go all-in. Bets are automatically clamped to your stack.

---

## poker_env.py — The Simulator

Supports **2–9 players**. Seat 0 is always the Button.

**Blind structure:** SB = half big blind. Default: 200 chip buy-in, BB = 2 (100 BB deep).

**Position assignment** is automatic based on table size:

```
2p: BTN  BB
3p: BTN  SB  BB
4p: BTN  SB  BB  CO
5p: BTN  SB  BB  HJ  CO
6p: BTN  SB  BB  UTG  HJ  CO
7p: BTN  SB  BB  UTG  LJ  HJ  CO
8p: BTN  SB  BB  UTG  UTG1  LJ  HJ  CO
9p: BTN  SB  BB  UTG  UTG1  UTG2  LJ  HJ  CO
```

**Betting rounds** play out in full. A street ends only once every live player has acted *and* matched the current bet:

- Checking passes the action along instead of ending the street — check/check is what advances a street, and the BB gets its option preflop.
- A raise reopens the action for everyone behind it.
- Folding ends the hand only when a single player is left; at a 9-handed table eight folds are needed for a walk.
- Preflop the action opens left of the BB (UTG); post-flop it opens left of the button, so the BTN closes every post-flop street.
- Raises are the bigger of the requested size and the last raise increment (minimum one BB), and are capped at `MAX_RAISES_PER_STREET` (4). The cap is what keeps the game tree finite for CFR.
- Players who run out of chips are marked all-in and skipped. If fewer than two players can still bet, the remaining board runs out to showdown.

**Bet tracking per player per street:**
- `current_bet` — the highest amount invested by any player this street (what you need to match to call)
- `street_invested` — how much each player has put in this street (resets each street)
- `total_invested_all` — cumulative chips invested across all streets (used for payoff calculation)
- `last_raise_size` / `raises_this_street` — drive the minimum raise and the raise cap

**Payoffs** are net (chips won minus chips invested) and always sum to zero across the table:
- Everyone folds but one: the last player standing takes the pot, so an uncalled bet comes back.
- Showdown: only players who did not fold are evaluated via `treys`. Ties chop.
- Side pots are built by splitting the pot at each all-in level, so a short stack can win only the portion it paid for.

**Other state tracked:**
- `facing_bet` — whether the player to act owes chips
- `folded` / `all_in` / `acted_this_street` — the live betting-round state
- `preflop_aggressor` — who last raised preflop (used for c-bet logic)
- `in_position_player` — whoever acts last post-flop (the BTN, or the nearest live seat before it)
- `went_to_showdown` — whether the hand was decided by cards rather than by folds

---

## hemo_bot.py — Position-Aware Strategy Bot

Models a deep-stacked low-stakes strategy. Ranges are keyed by **position name** so they scale automatically as players are added.

### Preflop Opening Ranges

Never limps — raise or fold only. Branching is driven by how many **raises**
have gone in, not by whether the player owes chips: the SB and the heads-up
button owe part of a blind in an unopened pot and open there rather than
treating it as a raise to defend against.

| Position | Approx % | Pairs | Key suited | Key offsuit |
|----------|-----------|-------|-----------|-------------|
| UTG2 | ~10% | 99+ | AJs+, KQs | AKo |
| UTG1 | ~12% | 88+ | ATs+, KQs, JTs | AQo+ |
| UTG  | ~14% | 77+ | A9s+, KTs+, QTs+, JTs | AQo+, KQo |
| LJ   | ~17% | 55+ | A5s+, KTs+, QTs+, JTs | AQo+, KQo |
| HJ   | ~22% | 55+ | A2s+, KTs+, QTs+, JTs | AJo+, KQo |
| CO   | ~29% | 22+ | A2s+, K9s+, Q9s+, J9s+, 65s+, T8s+ | AJo+, KQo, QJo, JTo |
| SB   | ~35% | 22+ | A2s+, K5s+, Q8s+, J8s+, 86s+, 76s, 65s | A2o+, K8o+, Q9o+, J9o+ |
| BTN  | ~48% | 22+ | A2s+, K2s+, Q5s+, J7s+, T7s+, 75s+ | A2o+, K8o+, Q9o+, J9o+, T9o |

### BB Defense

The BB is the one seat that does not use an opening range. It already has a
blind in and closes the action, so it continues on pot odds — far wider than
anyone opens. How wide depends on **who raised**, so the tiers are keyed to the
opener's position, not the BB's.

| Opener | Opens | BB defends | 3-bet | Call |
|--------|-------|-----------|-------|------|
| UTG / UTG1 / UTG2 / LJ | ~10–17% | **31%** | QQ+, AK, AQs, A5s–A2s | 22+, A2s+, K7s+, Q8s+, J8s+, T7s+, 54s+, A8o+, K9o+, Q9o+, J9o+, T9o |
| HJ | ~22% | **41%** | JJ+, AK, AQs, KQs, A5s–A2s | 22+, A2s+, K5s+, Q6s+, J6s+, T6s+, 43s+, A6o+, K8o+, Q8o+, J8o+, T8o+, 98o |
| CO | ~29% | **51%** | JJ+, AQ+, KQs, A5s–A2s | 22+, any suited ace or king, Q4s+, J5s+, T5s+, all connectors and one-gappers, A4o+, K7o+, Q7o+, J7o+, T7o+, 97o+, 87o |
| BTN (and the HU button) | ~48% | **57%** | TT+, AQ+, AJs, KQs, A5s–A2s, K7s–K5s | 22+, any suited ace, king or queen, J3s+, T3s+, all connectors and one-gappers, A2o+, K7o+, Q7o+, J7o+, T7o+, 96o+, 87o |
| SB | ~35% | **67%** | 99+, AQ+, AJs, KJs+, QJs, A5s–A2s | 22+, **every** suited hand, A2o+, K4o+, Q6o+, J6o+, T6o+, 96o+, 86o+, 76o |

Two things drive the shape:

- **The SB tier is the widest** even though the SB opens tighter than the BTN. It is the only spot where the BB has position — post-flop the SB acts first and the BB closes every street.
- **The 3-bet ranges are polarised**, not merely the top of the range: a value core plus the suited wheel aces (A2s–A5s). Those are the best bluffs because they block AA and AK and still flop nut-flush and wheel draws.

In an unraised pot the BB has the option and never folds — it raises TT+, AQ+, AJs+, KQs to punish limpers and checks everything else.

Frequencies are verified against solver-typical targets by `test_hemo_ranges.py`.

### Facing a 3-Bet Preflop

| Hand | In Position | Out of Position |
|------|-------------|-----------------|
| AA, KK | 4-bet (willing to jam) | 4-bet (willing to jam) |
| QQ, AK | 4-bet (not jam) | 4-bet (not jam) |
| Any pair, suited ace, AQo+, KJs+, KQo | Call | Fold |
| Everything else | Fold | Fold |

### Post-Flop

| Situation | Action |
|-----------|--------|
| Facing a bet, hand strength > 0.85 | Raise for value |
| Facing a bet, weaker hand | Fold — low-stakes players don't bluff |
| Hand strength > 0.75, any position | Bet/raise for value |
| In position + was preflop aggressor | C-bet |
| Everything else | Check |

---

## cfr_bot.py — CFR Bot

Uses **Counterfactual Regret Minimization** — the same core algorithm behind Libratus and Pluribus.

- Starts with uniform random strategy (33% fold/call/raise)
- After each simulated hand, tracks *regret*: how much better a different action would have been
- Shifts probability toward higher-regret actions over time
- The final strategy is the **average** across all iterations, which converges toward a Nash equilibrium

```bash
python3 train.py --iterations 10000
python3 play.py --cfr cfr_bot.pkl
```

---

## screen_reader.py — OpenCV Screen Reader

Captures a live poker client window and attempts to detect game state using computer vision and OCR.

**Step 1 — calibrate regions** (do this once with your poker client open):
```bash
python3 screen_reader.py --calibrate
```
Screenshots your screen after a 3-second delay, then you drag rectangles around each element. Saves to `screen_config.json`.

**Step 2 — live debug window:**
```bash
python3 screen_reader.py
python3 screen_reader.py --fps 4   # faster refresh
```
Opens a window with detected regions outlined and detected values overlaid. Press `q` to quit.

**Single snapshot:**
```bash
python3 screen_reader.py --once
```

**How detection works:**

| Element | Technique |
|---------|-----------|
| Card rectangles | Contour detection on light-colored regions |
| Card rank | Tesseract OCR on the top-left corner of each card |
| Card suit | HSV color analysis — red = hearts/diamonds, dark = spades/clubs |
| Pot / stacks | Tesseract OCR with digit-only whitelist, strips currency symbols |

**Known limitations:**
- Hearts vs diamonds and clubs vs spades can be ambiguous without card image templates — a future improvement would swap color analysis for template matching
- OCR accuracy depends on region tightness and font contrast — calibrate carefully
- The default regions in `DEFAULT_CONFIG` are placeholder values; calibration is required for your specific client

---

## Progression Path

| Stage | Goal |
|-------|------|
| 1 | Rule-based bot beats random opponents |
| 2 | hemobot: position-aware, proper bet sizing, stack-safe |
| 3 | CFR bot trained via self-play, reaches Nash equilibrium |
| 4 | Neural network trained on CFR strategy outputs |
| 5 | Full GTO approximation with hand abstraction |
