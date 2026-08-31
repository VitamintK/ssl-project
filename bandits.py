"""Abstract adversarial bandits (online learners) over a fixed set of arms.

Used by APSRO to choose, online, the opponent mixture that a policy trains against.
Feedback is partial (bandit): only the sampled arm's reward is observed. Rewards are
normalized from ``reward_range`` to [0, 1] internally so step sizes transfer across games.
"""
from abc import ABC, abstractmethod
import numpy as np


class AdversarialBandit(ABC):
    def __init__(self, num_arms: int, reward_range=(-1.0, 1.0), seed=None):
        if num_arms < 1:
            raise ValueError(f"num_arms must be >= 1, got {num_arms}")
        self.num_arms = num_arms
        self._lo, self._hi = float(reward_range[0]), float(reward_range[1])
        self._span = max(self._hi - self._lo, 1e-12)
        self.rng = np.random.RandomState(seed)

    def _normalize(self, reward: float) -> float:
        return float(np.clip((reward - self._lo) / self._span, 0.0, 1.0))

    @abstractmethod
    def distribution(self) -> np.ndarray:
        """Current mixture over arms: shape (num_arms,), non-negative, sums to 1."""

    @abstractmethod
    def update(self, arm: int, reward: float) -> None:
        """Observe ``reward`` (raw, in reward_range) for the sampled ``arm``."""

    def sample(self) -> int:
        return int(self.rng.choice(self.num_arms, p=self.distribution()))


class Hedge(AdversarialBandit):
    """Exponential weights with importance-weighted (Exp3-style) bandit updates."""

    def __init__(self, num_arms, reward_range=(-1.0, 1.0), eta=0.032, gamma=0.06, seed=None):
        super().__init__(num_arms, reward_range, seed)
        self.eta = eta
        self.gamma = gamma if num_arms > 1 else 0.0
        self.log_weights = np.zeros(num_arms)

    def distribution(self) -> np.ndarray:
        w = self.log_weights - self.log_weights.max()
        p = np.exp(w)
        p /= p.sum()
        return (1.0 - self.gamma) * p + self.gamma / self.num_arms

    def update(self, arm: int, reward: float) -> None:
        p = self.distribution()
        est = self._normalize(reward) / max(p[arm], 1e-12)  # unbiased reward estimate
        self.log_weights[arm] += self.eta * est


class RegretMatching(AdversarialBandit):
    """Regret matching (RM+) with mean-tracked counterfactual utilities (bandit feedback).

    Each arm's counterfactual utility is its running empirical mean reward (0.5, the middle
    of the normalized range, until first sampled). Cumulative regret accumulates
    (mean_i - realized_reward) each round and is clamped at 0 (RM+); the mixture is
    proportional to positive cumulative regret. This avoids the 1/p importance-weight
    variance explosion of naive bandit regret matching.
    """

    def __init__(self, num_arms, reward_range=(-1.0, 1.0), gamma=0.02, seed=None):
        super().__init__(num_arms, reward_range, seed)
        self.gamma = gamma if num_arms > 1 else 0.0
        self.cum_regret = np.zeros(num_arms)
        self.sum_reward = np.zeros(num_arms)
        self.count = np.zeros(num_arms)

    def _means(self) -> np.ndarray:
        # optimistic init: an unsampled arm is assumed max reward (1.0), so it gets explored.
        return np.where(self.count > 0, self.sum_reward / np.maximum(self.count, 1.0), 1.0)

    def distribution(self) -> np.ndarray:
        pos = np.maximum(self.cum_regret, 0.0)
        s = pos.sum()
        base = np.full(self.num_arms, 1.0 / self.num_arms) if s <= 1e-12 else pos / s
        return (1.0 - self.gamma) * base + self.gamma / self.num_arms

    def update(self, arm: int, reward: float) -> None:
        r = self._normalize(reward)
        self.sum_reward[arm] += r
        self.count[arm] += 1.0
        # instantaneous regret of each arm vs the reward actually obtained this round
        self.cum_regret = np.maximum(self.cum_regret + (self._means() - r), 0.0)


def make_bandit(name: str, num_arms: int, reward_range=(-1.0, 1.0), seed=None) -> AdversarialBandit:
    name = name.lower()
    if name == "hedge":
        return Hedge(num_arms, reward_range=reward_range, seed=seed)
    if name == "rm":
        return RegretMatching(num_arms, reward_range=reward_range, seed=seed)
    raise ValueError(f"unknown bandit {name!r}; expected 'hedge' or 'rm'")
