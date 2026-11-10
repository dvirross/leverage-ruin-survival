import numpy as np
import pytest

from ruin.games import simulate_doubling, simulate_doubling_at
from ruin.leverage import daily_rebalanced, margin_loan
from ruin.paths import longest_run, short_trade_outcomes
from ruin.windows import annualized_growth, sweep_daily, sweep_margin, window_starts


def make_series(n=600, seed=3):
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0004, 0.012, n)
    r[200] = -0.15   # a crash
    r[420] = -0.09
    rf = np.full(n, 0.0001)
    return r, rf


def test_window_starts():
    assert list(window_starts(100, 50, 21)) == [0, 21, 42]
    assert len(window_starts(10, 20)) == 0


@pytest.mark.parametrize("L", [0.5, 1.0, 2.0, 5.0])
@pytest.mark.parametrize("spread", [0.0, 0.015])
def test_sweep_daily_matches_single(L, spread):
    r, rf = make_series()
    starts = window_starts(len(r), 250, 37)
    out = sweep_daily(r, rf, starts, 250, L, spread)
    for i, s in enumerate(starts):
        path = daily_rebalanced(r[s:s + 250], rf[s:s + 250], L, spread)
        assert out["final"][i] == pytest.approx(path[-1])
        assert out["min"][i] == pytest.approx(min(path.min(), 1.0))


@pytest.mark.parametrize("L,m", [(2.0, 0.25), (4.0, 0.25), (6.0, 0.25), (-1.0, 0.30), (-2.0, 0.30), (1.25, 0.25)])
@pytest.mark.parametrize("action", ["close", "restore"])
@pytest.mark.parametrize("spread", [0.0, 0.015])
def test_sweep_margin_matches_single(L, m, action, spread):
    r, rf = make_series()
    starts = window_starts(len(r), 250, 37)
    out = sweep_margin(r, rf, starts, 250, L, m, spread, action)
    for i, s in enumerate(starts):
        res = margin_loan(r[s:s + 250], rf[s:s + 250], L, m, spread, action)
        assert out["final"][i] == pytest.approx(res.equity[-1])
        assert out["min"][i] == pytest.approx(res.equity.min())
        assert out["ncalls"][i] == len(res.calls)
        assert out["first_call"][i] == (res.calls[0] if res.calls else -1)
        assert bool(out["wiped"][i]) == (res.wiped_out_at is not None)


def test_annualized_growth():
    g = annualized_growth(np.array([2.0, 0.0, 1.0]), 252)
    assert g[0] == pytest.approx(1.0) and g[1] == -1.0 and g[2] == pytest.approx(0.0)


def test_simulate_doubling_at_matches_full():
    full = simulate_doubling(1, 100, 0.5, 300, 50, np.random.default_rng(1))
    at = simulate_doubling_at(1, 100, 0.5, [10, 300], 50, np.random.default_rng(1))
    assert np.allclose(at[10], full[:, 10]) and np.allclose(at[300], full[:, 300])


def test_longest_run():
    assert longest_run(np.array([0, 1, 1, 0, 1, 1, 1, 0], bool)) == (3, 4)
    assert longest_run(np.array([0, 0], bool)) == (0, -1)


def test_short_trade_outcomes():
    # price rises to 1.5 then returns to 1.0 on day 4; second start never under water
    p = np.array([1.0, 1.2, 1.5, 1.1, 1.0, 0.8, 0.7])
    o = short_trade_outcomes(p, [0, 4], 4)
    assert o["under_water"][0] and o["payoff_day"][0] == 4 and o["mae"][0] == pytest.approx(0.5)
    assert not o["under_water"][1] and o["payoff_day"][1] == -1 and o["mae"][1] == 0.0
    # never pays off within the window
    o2 = short_trade_outcomes(p, [0], 3)
    assert o2["under_water"][0] and o2["payoff_day"][0] == -1 and o2["mae"][0] == pytest.approx(0.5)


def test_short_trade_outcomes_min_adverse():
    p = np.array([1.0, 1.05, 1.0, 1.4, 1.2, 0.9])
    o0 = short_trade_outcomes(p, [0], 5)
    assert o0["payoff_day"][0] == 2                      # back to entry after the 5% rise
    o = short_trade_outcomes(p, [0], 5, min_adverse=0.2)
    assert o["under_water"][0] and o["payoff_day"][0] == 5 and o["mae"][0] == pytest.approx(0.4)
    o2 = short_trade_outcomes(p, [0], 5, min_adverse=0.5)
    assert not o2["under_water"][0]
