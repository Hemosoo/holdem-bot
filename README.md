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

## Eight-seat table reader

`table_reader.py` reads screenshots, video, or live frames of the supplied table
skin and emits JSON. It uses OCR plus isolated glyph templates calibrated from
the original images in `fixtures/`. It never infers face-down cards.

```bash
source .venv/bin/activate
python table_reader.py --image "fixtures/ChatGPT Image Nov 11, 2026, 10_12_15 PM.png" --confirmations 1 --overlay /tmp/table-overlay.png
python table_reader.py --video /path/session.mp4 > /tmp/observations.jsonl
python table_reader.py --live --layout expanded > /tmp/observations.jsonl
python evaluate_table_reader.py fixtures --output /tmp/table-evaluation
python -m unittest -v test_table_reader
```

Use `--viewport X Y WIDTH HEIGHT` when the client is inside a larger screenshot.
The viewport must frame the whole client as in the supplied examples. Auto layout
uses aspect ratio: compact has a 1024x677 reference, expanded 1024x765. It does not
track a moving window or discover arbitrary seat arrangements. Inspect an overlay
before relying on readings. Live capture requires an active desktop.

Output:
- `fields`: observed value, heuristic confidence, raw OCR and reference-frame box.
- `confirmed`: values repeated across two processed frames by default. A missing
  observation removes confirmation immediately. A static screenshot needs
  `--confirmations 1` to populate this map.
- `table`: eight players, board, pot, dealer seat and inferred street. This is a
  summary of the current observations, **not** a confirmed game state. Seats run
  clockwise from upper-left; the bottom seat is 5.
- `warnings`: consistency problems. Rejected duplicate cards retain `candidate`
  for review, while `value` remains null.

Players include name, stack, inward bet, recognized status labels, a positive
blue-turn-ring observation, and exposed cards. Missing is unknown, not zero,
folded, absent, or preflop. Three to five consecutive identified board cards can
establish flop/turn/river. Amounts use displayed BB units. Confidence is not a
calibrated probability; raw OCR and overlays remain essential for review.

### Validation and remaining limits

On the two original calibration images, 36 of 37 labeled checks match. All five
board cards, hero 3c/7h, both pots, both dealers, four visible bets, the blue turn
ring and 15 stacks match. The compact upper-left stack is left unknown because
OCR disagrees between 211 and 21. The evaluator intentionally exits nonzero for
that mismatch and writes a complete report rather than hiding it.

The expanded frame's exposed pair visually contains two 7s cards. Both visual
candidates are retained and rejected as a duplicate; no valid pair is invented.
RAISE, Waiting and SITTING OUT labels have regression coverage. Several names
still contain OCR errors or remain unknown; names are not covered by the 37-field
accuracy count. Seat numbers provide stable identifiers within this layout.

These are calibration results, not held-out accuracy. Glyphs came from the same
images (see `vision_templates/README.md`). Board-card tests also exercise dimming
and downsampling, but there is no validation on actual motion, animations,
new camera framing, unseen fonts, or new table skins. Tilted/dim cards are only
calibrated at the displayed positions. Diamonds and unrepresented ranks rely on
font/OCR fallbacks and need further real examples. OCR is synchronous and the
actual processing interval includes computation time; this is not a verified
real-time reader. No automatic game actions are performed.

Dependencies use the existing project environment. Tesseract must be installed
(`brew install tesseract` on macOS). DejaVu Sans is bundled in `assets/fonts/`
with its license, so suit matching does not depend on a Linux system-font path.

### Recording layout and video regression checks

The supplied 71.38-second, 2322x1552, 60 fps test recording uses a different layout.
Select it explicitly; `auto` still selects the two original screenshot layouts.

```bash
python table_reader.py --video /path/pokergame_c9qXwyb6.mp4 --layout video --interval 5 > /tmp/video-observations.jsonl
python evaluate_table_video.py /path/pokergame_c9qXwyb6.mp4 --output /tmp/video-validation
```

Video layout uses a 1024x684 reference, locates board-card rectangles on the felt,
and uses the chip icon to locate the pot and hero stack's first digit. This avoids
clipping when the amount panel shifts with text length. Hero cards use the new
positions. These changes do not provide arbitrary window tracking.

CLI video results include source video seconds and frame index, separately from
wall-clock observation time. The evaluator checks nine labeled instants: 0, 5,
10, 20, 30, 40, 50, 60 and 70 seconds. Each checks five board slots, two hero-card
slots, pot and hero stack (81 comparisons total). It writes a mismatch report,
observations and overlays, and fails on mismatches or decoding errors. This is
sampled-frame evaluation, not full-frame or complete event coverage. Opponent
names, opponent stacks, every bet and every animation are not validated by this
81-field comparison. Frames at 40s and 50s provided additional glyph calibration;
other sample frames use the same recording and are not an independent dataset.

Retained normalized frames `fixtures/video-40.png` and `video-70.png` exercise a
flop, hero cards, leading digits of amounts, and removal of stale board/card
confirmations at a hand transition. OCR takes roughly 6–7 seconds per frame on
the current machine. The live/video path remains an offline diagnostic tool at
this speed, not a real-time observer. `--interval` controls source sampling for
video; it does not make OCR run faster.

Latest labeled video run: **81/81 comparisons match across the nine sampled
frames**. This includes removal of both hero cards and all board cards at 70s.
The evaluator's detailed report and overlays are under
`/workspace/.holdem-setup/video-validation/` in the current environment. The
original-still calibration continues to leave the ambiguous 211/21 stack unknown.
Video-specific glyph templates are enabled only with `--layout video` so their
additional variants cannot make still-image suit matching ambiguous.

## Small table dashboard

```bash
source .venv/bin/activate
python dashboard.py                  # starts with the retained test frame
python dashboard.py --video /path/recording.mp4 --layout video
python dashboard.py --live --layout video --viewport X Y WIDTH HEIGHT
python -m unittest -v test_dashboard
```

The local server binds only to 127.0.0.1 on port 8765 (change with `--port`). Open
that port in a browser on the same machine. It does not require Flask or a new
package. Use Read frame for a screenshot/live source or enter a source time for a
recording. Open image uploads a frame up to 16 MiB. Auto update requests a new
reading when the previous one has completed; for recordings it advances five
source seconds each time. Capture latency remains about 6–7 seconds. A failed
capture disables suggestions even if an older image is still displayed.

The dashboard shows community and hero cards, eight player rows, stacks, visible
bets, statuses, dealer-relative positions and the detected hero turn ring. Choose
your seat (default 5, bottom seat), verify the dealer and check exactly the seats
dealt into this hand. Folded players remain checked; exclude empty/sitting-out
seats. Positions rotate clockwise; BTN means dealer and CO means cutoff. Heads-up
shows BTN / SB. Unreadable names and amounts remain uncertain; no hidden card is
inferred. A missing turn ring is unknown, not proof that it is someone else's turn.

Suggestions adapt the existing **hemobot rules**, not a solver or a claim of the
optimal move. Confirm the street if it cannot be inferred, amount to call (0 for a
free check), raise count preflop, preflop aggressor when relevant, whether you act
last postflop, and whether raising is allowed. The reader cannot reliably recover
all that betting history from a screenshot. Confirm context for the current frame;
a new frame invalidates confirmation so old amounts cannot carry into a new hand.
A reading older than 30 seconds cannot produce advice. Automatic turn detection
uses a current blue-ring observation; the user can override it for tests.

Incomplete or contradictory cards, unknown hero stack, uncertain positions, missing
context, or an unconfirmed frame withhold recommendations. No bet size is computed
and no game action is executed. The UI shows the policy source, source time, capture
age and uncertainty. It can analyze a test recording independently of any game.

Verification: six dashboard policy tests cover positions, heads-up, excluded seats,
turn/stale/confirmation gates, missing or duplicate cards, invalid input and policy
integration. Browser checks using Chromium cover eight rows, the sample frame's
HJ position, a confirmed Fold suggestion, dealer rotation to LJ, turn gating and
a narrow mobile viewport with no JavaScript errors.


### macOS startup font fix

If an older checkout reports `Suit templates require DejaVuSans.ttf`, pull the
latest `main`. The matcher and tests now load the bundled font relative to their
source file, so no system font installation is needed and startup works from any
working directory. Keep the `assets/fonts/` directory with the application.

### Live screen capture and macOS troubleshooting

Start live mode with `python dashboard.py --live --layout video`, keep the process
running, and open the dashboard in your browser. **Auto update is enabled by
default in live mode while the page remains open.** Each capture is followed by
OCR and a short display interval before the next capture; this is not video-rate
recognition. Uncheck Auto update to pause and inspect/confirm a frame.

The captured image is visible immediately, before OCR finishes. Progress shows
capturing vs recognition and elapsed seconds. Last completed readings stay visible
during updates, but suggestions are withheld while a fresh reading is pending.
If recognition fails (for example missing Tesseract), the captured preview remains
visible and the error appears in both the dashboard and Terminal.

Keep the game visible beside the dashboard or on another display. A covered game
window cannot be read from a screenshot. Drag around the entire game client in
the preview, including the player panels, after a reading finishes. The crop is
applied to that same image without taking another screenshot; subsequent live
captures reuse the selected area. Use entire image clears it. This is a fixed
screen rectangle: reselect it if the game window moves or resizes. It is not
automatic window tracking. `--monitor 2` selects a second display (default: 1).

On macOS, enable Screen Recording (or Screen & System Audio Recording) for the
application running Python under System Settings > Privacy & Security. That may
be Terminal, iTerm or VS Code. Quit and reopen that application after enabling it,
then restart the dashboard. A preview showing only wallpaper or the wrong app
means the game is not being captured; check permission, display and visibility.
The current capture code uses `mss.MSS()` rather than the deprecated `mss.mss()`.
An MSS deprecation warning alone does not establish whether capture succeeded.

### Off-center game windows and browser window sharing

If the game is beside the dashboard, do not treat the entire desktop as the game
client. The dashboard now offers **Share game window**: select the actual game
window in the browser picker. Frames from that selected stream are sent to the
local server for OCR, and Auto update starts automatically. No audio is requested.
Stop sharing (or the browser's stop-sharing control) releases the stream and
pauses updates. Use a recent Chrome browser on localhost; permission and available
window choices are controlled by the browser/OS. Keep the game unminimized.
This route does not require the dashboard's native MSS capture to see through the
browser. If you choose the entire screen instead, select the game crop afterward.

Switching to a browser stream clears a crop from the previous desktop source once;
subsequent streamed frames preserve a user-selected crop. Starting a crop drag
pauses Auto update so the preview cannot be replaced midway through selection.

Native `--live` mode without an explicit viewport now tries to locate the known
client using static rail/UI features, independently of its desktop position. This
conservative matcher was tested with the retained frames shifted to the left,
resized and placed beside a simulated second window. It checks distributed feature
matches and rectangular geometry. If it cannot confidently locate the client, it
shows an error and requests a manual crop or browser window share; it does not
silently reinterpret the whole desktop. It is not a universal game/window detector.
A manually selected crop disables automatic location and stays fixed. Resetting it
re-enables automatic location in native live mode. No automatic location is needed
for a shared game window already framed to the client.

The status text distinguishes no table amounts recognized from a table with no
recognized cards. Neither condition proves a hand ended; old cards are not carried
into a new reading as if they were current. The last completed snapshot may still
be displayed while the next one is processing, and is labeled as such.

Seat indices refer to the eight layout slots, not the number of occupied seats.
For the supplied five-player example, slots 0, 2, 3, 5 and 6 are occupied. With
dealer 0, these map to BTN, SB, BB, UTG and CO respectively. Confirm occupied seats
in the form; missing OCR by itself does not establish that a seat is empty.

### Export an incorrect reading for diagnosis

When a captured game is visible but fields are wrong, pause Auto update and wait
for recognition to finish, then click **Download diagnostic ZIP** in the preview
section. Review the screenshots before attaching the ZIP in chat: they contain
captured screen content, which can include other visible windows.

The export contains pixel-exact `captured-frame.png`, `table-crop.png`, the
normalized `recognition-overlay.png`, `readings.json` (including unknown values,
raw OCR and confidence), and `diagnostic.json` with the matched crop, layout,
source and capture timestamp. It does not include credentials or the process
environment. If a newer image was captured after the exported reading, it is
separately named `latest-capture-not-matched-to-readings.png`; it must not be used
as the ground truth for that older result. Exporting a diagnostic downloads it
locally; it does not upload or send any data automatically.

Recognition is still calibrated to the retained layouts and glyph examples. A
successful screen share does not by itself establish correct card recognition.
The diagnostic is needed to distinguish a crop/alignment issue, font/glyph
mismatch, occlusion or OCR error on a user's actual frames.
