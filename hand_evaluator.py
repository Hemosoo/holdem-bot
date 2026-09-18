from treys import Card, Evaluator

evaluator = Evaluator()


def evaluate_hand_strength(hole_cards: list[str], community_cards: list[str]) -> float:
    """
    Returns a normalized hand strength score between 0.0 (worst) and 1.0 (best).
    hole_cards and community_cards are lists of strings like ['Ah', 'Kd'].
    """
    if not community_cards:
        return 0.5  # pre-flop: no board to evaluate against

    board = [Card.new(c) for c in community_cards]
    hand = [Card.new(c) for c in hole_cards]

    score = evaluator.evaluate(board, hand)
    # treys scores: 1 (best, royal flush) to 7462 (worst, high card)
    # normalize to 0.0 (worst) -> 1.0 (best)
    return 1.0 - (score - 1) / 7461


def describe_hand(hole_cards: list[str], community_cards: list[str]) -> str:
    board = [Card.new(c) for c in community_cards]
    hand = [Card.new(c) for c in hole_cards]
    score = evaluator.evaluate(board, hand)
    rank_class = evaluator.get_rank_class(score)
    return evaluator.class_to_string(rank_class)
