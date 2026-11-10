import numpy as np
import pytest

from ruin.leverage import approx_growth, daily_rebalanced, kelly_leverage, margin_loan


def test_daily_rebalanced_wipeout_threshold():
    r = np.array([0.01, -0.2, 0.05])
    rf = np.zeros(3)
    assert daily_rebalanced(r, rf, 5.0)[-1] == 0.0  # 5 x -20% = -100%
    assert daily_rebalanced(r, rf, 4.0)[-1] > 0.0


def test_daily_rebalanced_l1_is_buy_and_hold():
    r = np.array([0.1, -0.05, 0.02])
    path = daily_rebalanced(r, np.full(3, 0.001), 1.0)
    assert path[-1] == pytest.approx(np.prod(1 + r))


def test_financing_cost():
    r = np.zeros(252)
    rf = np.full(252, 0.0001)
    path = daily_rebalanced(r, rf, 2.0, spread_annual=0.0252)
    assert path[-1] == pytest.approx((1 - 0.0001 - 0.0001) ** 252)


def test_margin_loan_close_on_call():
    # L=2: V=2, D=1. A 40% fall gives V=1.2, E=0.2, E/V=0.167 < 0.25 -> call and close
    res = margin_loan(np.array([-0.4, 0.5]), np.zeros(2), 2.0, maintenance=0.25, action="close")
    assert res.calls == [1] and res.closed_at == 1
    assert res.equity[1] == pytest.approx(0.2)
    assert res.equity[2] == pytest.approx(0.2)  # out of the market, misses the rebound


def test_margin_loan_restore_locks_loss():
    res = margin_loan(np.array([-0.4, 0.5]), np.zeros(2), 2.0, maintenance=0.25, action="restore")
    assert res.equity[2] == pytest.approx(0.2 * (1 + 2 * 0.5))


def test_margin_loan_no_call_small_moves():
    res = margin_loan(np.array([-0.1, 0.1]), np.zeros(2), 2.0)
    assert res.calls == [] and res.equity[-1] == pytest.approx(2 * 0.9 * 1.1 - 1)


def test_margin_loan_gap_wipeout():
    res = margin_loan(np.array([-0.6]), np.zeros(1), 2.0)
    assert res.wiped_out_at == 1 and res.equity[-1] == 0.0


def test_short_margin_call():
    # short 1 unit per unit equity: C=2, S=1. Price +70%: S=1.7, E=0.3, E/S=0.176 -> call
    res = margin_loan(np.array([0.7]), np.zeros(1), -1.0, maintenance=0.3)
    assert res.closed_at == 1 and res.equity[1] == pytest.approx(0.3)


def test_short_gains_when_price_falls():
    res = margin_loan(np.array([-0.5]), np.zeros(1), -1.0)
    assert res.equity[-1] == pytest.approx(1.5)


def test_kelly_maximizes_approx_growth():
    mu, r, sigma = 0.09, 0.03, 0.18
    Ls = np.linspace(0, 5, 5001)
    g = [approx_growth(L, mu, r, sigma) for L in Ls]
    assert Ls[int(np.argmax(g))] == pytest.approx(kelly_leverage(mu - r, sigma), abs=1e-3)
