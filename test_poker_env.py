"""
Betting-round tests for poker_env.

    python3 test_poker_env.py
"""

import random
import sys

from poker_env import *
import poker_env as E

fails = []
def check(name, cond, extra=""):
    print(("  ok   " if cond else "  FAIL ") + name + ("" if cond else f"   {extra}"))
    if not cond: fails.append(name)

print("\n[1] heads-up: BB gets its option after a limp")
random.seed(3)
s = new_game(2)
check("BTN acts first preflop", s.current_player == 0, s.current_player)
apply_action(s, CALL)
check("still preflop after limp", s.street == "preflop", s.street)
check("BB to act", s.current_player == 1, s.current_player)
check("no flop yet", s.community == [], s.community)
check("BB may check not fold", legal_actions(s) == [CALL, RAISE], legal_actions(s))
apply_action(s, CALL)
check("flop dealt after option checked", s.street == "flop" and len(s.community) == 3, s.street)
check("BB first to act on flop", s.current_player == 1, s.current_player)

print("\n[2] check-check advances exactly one street")
apply_action(s, CALL)
check("still flop after one check", s.street == "flop", s.street)
check("BTN to act", s.current_player == 0, s.current_player)
apply_action(s, CALL)
check("turn after check-check", s.street == "turn" and len(s.community) == 4, s.street)

print("\n[3] bet / call closes the street, bet / raise / call reopens")
apply_action(s, RAISE, 6)
check("bettor's raise counted", s.raises_this_street == 1, s.raises_this_street)
check("still turn, opponent owes", s.street == "turn" and s.facing_bet, (s.street, s.facing_bet))
apply_action(s, RAISE, 12)
check("re-raise keeps street open", s.street == "turn", s.street)
check("action reopened for first bettor", s.current_player == 1, s.current_player)
apply_action(s, CALL)
check("river after call closes turn", s.street == "river", s.street)

print("\n[4] 6-max turn order")
random.seed(7)
s6 = new_game(6)
check("UTG (seat 3) opens preflop", s6.current_player == 3, s6.current_player)
check("BTN is in position", s6.in_position_player == 0, s6.in_position_player)
order = []
while s6.street == "preflop" and not s6.terminal:
    order.append(s6.current_player)
    apply_action(s6, CALL)
check("preflop order UTG->BB", order == [3, 4, 5, 0, 1, 2], order)
check("flop reached", s6.street == "flop", s6.street)
check("SB (seat 1) opens post-flop", s6.current_player == 1, s6.current_player)
post = []
while s6.street == "flop" and not s6.terminal:
    post.append(s6.current_player)
    apply_action(s6, CALL)
check("post-flop order SB->BTN", post == [1, 2, 3, 4, 5, 0], post)

print("\n[5] one fold does not end a 6-handed hand")
random.seed(11)
s6 = new_game(6)
apply_action(s6, FOLD)
check("hand continues", not s6.terminal, s6.terminal)
check("folder recorded", s6.folded == {3}, s6.folded)
check("action passed to seat 4", s6.current_player == 4, s6.current_player)
for _ in range(3):
    apply_action(s6, FOLD)                   # seats 4, 5, 0 fold
check("still live with SB and BB", not s6.terminal, (s6.terminal, s6.folded))
apply_action(s6, FOLD)          # SB folds, BB alone
check("hand ends with one player", s6.terminal, s6.terminal)
check("no showdown on a walk", not s6.went_to_showdown)
check("BB wins the blinds", abs(s6.returns[2] - 1.0) < 1e-9, s6.returns)
check("returns sum to zero", abs(sum(s6.returns.values())) < 1e-9, sum(s6.returns.values()))

print("\n[6] folded players are excluded from showdown")
random.seed(23)
s3 = new_game(3)
apply_action(s3, FOLD)                       # BTN folds preflop
while not s3.terminal:
    apply_action(s3, CALL)
check("went to showdown", s3.went_to_showdown)
check("BTN folded having posted nothing", abs(s3.returns[0]) < 1e-9, s3.returns)
check("only a blind poster can be the winner", max(s3.returns, key=s3.returns.get) in (1, 2), s3.returns)
check("returns sum to zero", abs(sum(s3.returns.values())) < 1e-9, s3.returns)

print("\n[6b] a folder who already invested forfeits it at showdown")
random.seed(23)
s3b = new_game(3)
apply_action(s3b, RAISE, 6)                  # BTN opens
apply_action(s3b, CALL)                      # SB calls
apply_action(s3b, CALL)                      # BB calls
apply_action(s3b, CALL)                      # SB checks flop
apply_action(s3b, CALL)                      # BB checks flop
apply_action(s3b, RAISE, 20)                 # BTN bets flop
apply_action(s3b, FOLD)                      # SB folds 8 already in
before = s3b.total_invested_all[1]
while not s3b.terminal:
    apply_action(s3b, CALL)
check("folded SB forfeits exactly what it put in", abs(s3b.returns[1] + before) < 1e-9, (s3b.returns, before))
check("SB is not paid despite showdown", 1 not in [p for p in s3b.returns if s3b.returns[p] > 0], s3b.returns)
check("returns sum to zero", abs(sum(s3b.returns.values())) < 1e-9, s3b.returns)

print("\n[7] minimum raise is enforced")
random.seed(5)
s = new_game(2)
apply_action(s, RAISE, 0.5)                  # asks for a tiny raise
check("bumped to a legal raise", s.current_bet >= 2 + 2 - 1e-9, s.current_bet)

print("\n[8] raise cap stops the betting war")
random.seed(5)
s = new_game(2)
n = 0
while RAISE in legal_actions(s) and s.street == "preflop" and not s.terminal:
    apply_action(s, RAISE, 4); n += 1
    if n > 20: break
check("cap reached", s.raises_this_street >= E.MAX_RAISES_PER_STREET or s.terminal, (n, s.raises_this_street))
check("raise no longer legal", s.terminal or RAISE not in legal_actions(s), legal_actions(s))

print("\n[9] all-in short stack: side pot, and the board runs out")
random.seed(31)
s = new_game(3, starting_stack=200.0)
s.stacks[0] = 20.0                           # BTN is short
while not s.terminal:
    la = legal_actions(s)
    apply_action(s, RAISE if RAISE in la else CALL, 400)
check("reached showdown", s.went_to_showdown, (s.street, s.terminal))
check("full board dealt", len(s.community) == 5, s.community)
check("returns sum to zero", abs(sum(s.returns.values())) < 1e-9, s.returns)
check("short stack cannot lose over 20", s.returns[0] >= -20 - 1e-9, s.returns)
check("short stack cannot win over 40", s.returns[0] <= 40 + 1e-9, s.returns)

print("\n[10] fuzz: random legal actions, 2-9 players")
random.seed(99)
bad = []
for trial in range(4000):
    n = random.randint(2, 9)
    stack = random.choice([20.0, 50.0, 200.0])
    s = new_game(n, starting_stack=stack)
    steps = 0
    while not s.terminal:
        la = legal_actions(s)
        if not la:
            bad.append(("no legal actions", trial)); break
        apply_action(s, random.choice(la), random.choice([2.0, 6.0, 25.0, 500.0]))
        steps += 1
        if steps > 400:
            bad.append(("did not terminate", trial, n)); break
    if s.terminal:
        if abs(sum(s.returns.values())) > 1e-6:
            bad.append(("returns nonzero", trial, sum(s.returns.values())))
        if abs(sum(s.stacks.values()) - n * stack) > 1e-6:
            bad.append(("chips not conserved", trial, sum(s.stacks.values()), n * stack))
        if len(s.returns) != n:
            bad.append(("missing returns", trial, s.returns))
        for p in s.stacks:
            if s.stacks[p] < -1e-9:
                bad.append(("negative stack", trial, p, s.stacks[p]))
        if s.went_to_showdown and len(s.community) != 5:
            bad.append(("showdown without full board", trial, s.community))
check("4000 random hands clean", not bad, bad[:5])

print("\n[11] chopped pot splits evenly")
random.seed(1)
s = new_game(2)
s.hole_cards[0] = ["Ah", "Kh"]
s.hole_cards[1] = ["Ad", "Kd"]
s.deck = ["2c", "5s", "9h", "Jc", "Qs"] + s.deck
while not s.terminal:
    apply_action(s, CALL)
check("board is a chop", s.went_to_showdown)
check("both net zero", abs(s.returns[0]) < 1e-9 and abs(s.returns[1]) < 1e-9, s.returns)

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILURES: {fails}"))
sys.exit(1 if fails else 0)
