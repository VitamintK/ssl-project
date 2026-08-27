import numpy as np
import pytest
from bandits import AdversarialBandit, Hedge, RegretMatching, make_bandit


def _run(bandit, means, n=4000, seed=0):
    """Drive a bandit against fixed per-arm reward means (bandit feedback)."""
    rng = np.random.RandomState(seed)
    for _ in range(n):
        arm = bandit.sample()
        reward = means[arm] + 0.01 * rng.randn()
        bandit.update(arm, reward)
    return bandit.distribution()


@pytest.mark.parametrize("cls", [Hedge, RegretMatching])
def test_distribution_is_valid_simplex(cls):
    b = cls(num_arms=4, reward_range=(-1.0, 1.0), seed=1)
    for _ in range(50):
        b.update(b.sample(), 0.3)
    d = b.distribution()
    assert d.shape == (4,)
    assert np.all(d >= -1e-9)
    assert abs(d.sum() - 1.0) < 1e-6


@pytest.mark.parametrize("cls", [Hedge, RegretMatching])
def test_concentrates_on_best_arm(cls):
    means = np.array([0.9, 0.1, -0.2, 0.0])  # arm 0 is best
    d = _run(cls(num_arms=4, reward_range=(-1.0, 1.0), seed=0), means)
    assert int(np.argmax(d)) == 0
    assert d[0] > 0.5


@pytest.mark.parametrize("cls", [Hedge, RegretMatching])
def test_single_arm_is_degenerate_safe(cls):
    b = cls(num_arms=1, reward_range=(-1.0, 1.0), seed=0)
    assert b.sample() == 0
    b.update(0, 0.5)
    assert abs(b.distribution()[0] - 1.0) < 1e-9


def test_make_bandit_maps_names():
    assert isinstance(make_bandit("hedge", 3), Hedge)
    assert isinstance(make_bandit("rm", 3), RegretMatching)
    with pytest.raises(ValueError):
        make_bandit("nope", 3)
