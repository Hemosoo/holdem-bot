from hand_evaluator import evaluate_hand_strength


def decide_action(
    hole_cards: list[str],
    community_cards: list[str],
    legal_actions: list[int],
    pot: float = 0,
    stack: float = 0,
) -> int:
    """
    Simple rule-based decision engine.
    Returns an index into legal_actions.

    Thresholds:
      strength > 0.8  -> raise  (action index 2 if available, else call)
      strength > 0.5  -> call   (action index 1 if available, else fold)
      otherwise       -> fold   (action index 0)
    """
    strength = evaluate_hand_strength(hole_cards, community_cards)

    if strength > 0.8:
        preferred = [2, 1, 0]  # raise > call > fold
    elif strength > 0.5:
        preferred = [1, 0]     # call > fold
    else:
        preferred = [0]        # fold

    for action in preferred:
        if action in legal_actions:
            return action

    return legal_actions[0]
