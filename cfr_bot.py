import numpy as np
from collections import defaultdict

NUM_ACTIONS = 3  # 0=fold, 1=call, 2=raise
ACTION_NAMES = ["fold", "call", "raise"]


class CFRBot:
    def __init__(self):
        self.regret_sum: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(NUM_ACTIONS))
        self.strategy_sum: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(NUM_ACTIONS))

    def get_strategy(self, info_set: str) -> np.ndarray:
        regrets = self.regret_sum[info_set]
        positive = np.maximum(regrets, 0)
        total = positive.sum()
        if total > 0:
            return positive / total
        return np.ones(NUM_ACTIONS) / NUM_ACTIONS  # uniform if no positive regret

    def get_action(self, info_set: str, legal_actions: list[int]) -> int:
        strategy = self.get_strategy(info_set)
        # mask to legal actions only
        mask = np.zeros(NUM_ACTIONS)
        for a in legal_actions:
            if a < NUM_ACTIONS:
                mask[a] = 1.0
        masked = strategy * mask
        total = masked.sum()
        if total == 0:
            return legal_actions[0]
        probs = masked / total
        return int(np.random.choice(NUM_ACTIONS, p=probs))

    def update_regrets(self, info_set: str, action_idx: int, regret: float) -> None:
        self.regret_sum[info_set][action_idx] += regret
        self.strategy_sum[info_set][action_idx] += self.get_strategy(info_set)[action_idx]

    def get_average_strategy(self, info_set: str) -> np.ndarray:
        total = self.strategy_sum[info_set].sum()
        if total > 0:
            return self.strategy_sum[info_set] / total
        return np.ones(NUM_ACTIONS) / NUM_ACTIONS
