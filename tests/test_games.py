import numpy as np
import pytest

from ruin.games import doubling_cycle_outcomes, gamblers_ruin_probability, simulate_doubling


def test_fair_game_ruin_is_linear():
    assert gamblers_ruin_probability(30, 100, 0.5) == pytest.approx(0.7)


def test_ruin_formula_matches_simulation():
    rng = np.random.default_rng(1)
    capital, target, p, n = 10, 20, 0.47, 40_000
    x = np.full(n, capital)
    active = np.ones(n, bool)
    while active.any():
        step = np.where(rng.random(active.sum()) < p, 1, -1)
        x[active] += step
        active = (x > 0) & (x < target)
    sim = (x == 0).mean()
    se = np.sqrt(sim * (1 - sim) / n)
    assert abs(sim - gamblers_ruin_probability(capital, target, p)) < 4 * se


def test_ruin_formula_large_target_no_overflow():
    v = gamblers_ruin_probability(50, 5000, 0.49)
    assert 0.999 < v <= 1.0


def test_doubling_fair_expectation_zero():
    out = doubling_cycle_outcomes(1, 1023, 0.5)
    assert out["bets_affordable"] == 10
    assert out["loss_if_lost"] == 1023
    assert out["expected"] == pytest.approx(0.0, abs=1e-12)
    assert out["p_win"] == pytest.approx(1 - 2**-10)


def test_doubling_unfair_expectation_negative():
    assert doubling_cycle_outcomes(1, 1023, 18 / 37)["expected"] < 0


def test_simulated_doubling_mean_close_to_start_in_fair_game():
    rng = np.random.default_rng(7)
    paths = simulate_doubling(1, 100, 0.5, 200, 20_000, rng)
    final = paths[:, -1]
    se = final.std(ddof=1) / np.sqrt(len(final))
    assert abs(final.mean() - 100) < 4 * se
    assert (final >= 0).all()


def test_doubling_short_horizon_mostly_wins_long_horizon_mostly_loses():
    rng = np.random.default_rng(11)
    short = simulate_doubling(1, 100, 0.5, 10, 20_000, rng)[:, -1]
    long = simulate_doubling(1, 100, 0.5, 2_000, 20_000, rng)[:, -1]
    assert (short > 100).mean() > 0.75         # most players are ahead after a few bets
    assert (long < 1).mean() > 0.5               # most are ruined after many cycles, though the mean stays near 100
