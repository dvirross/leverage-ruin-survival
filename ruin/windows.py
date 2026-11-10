"""Vectorised sweeps of leverage strategies over many start dates (rolling windows of a return series).

``sweep_daily`` and ``sweep_margin`` give, for every window start, the same results as running
``leverage.daily_rebalanced`` / ``leverage.margin_loan`` on that window one by one (tests check this),
but with all windows advanced in parallel. Windows are r[s : s + n_days] for s in ``starts``.
"""
from __future__ import annotations

import numpy as np

from .leverage import TRADING_DAYS


def window_starts(n_obs: int, n_days: int, step: int = 21) -> np.ndarray:
    """Start indices 0, step, 2*step, ... of every window of n_days that fits in n_obs observations."""
    return np.arange(0, n_obs - n_days + 1, step)


def sweep_daily(r, rf, starts, n_days: int, L: float, spread_annual: float = 0.0) -> dict:
    """Daily-rebalanced leverage L over each window: final equity, minimum equity along the path (relative to
    the start, including the start value 1), and the arithmetic mean of daily equity returns."""
    r = np.asarray(r, float)
    rf = np.asarray(rf, float)
    s = spread_annual / TRADING_DAYS if L > 1 else 0.0
    idx = np.asarray(starts)[:, None] + np.arange(n_days)[None, :]
    eq = L * r[idx] - (L - 1.0) * (rf[idx] + s)
    path = np.cumprod(np.maximum(1.0 + eq, 0.0), axis=1)
    return {"final": path[:, -1], "min": np.minimum(path.min(axis=1), 1.0), "mean_daily": eq.mean(axis=1)}


def sweep_margin(r, rf, starts, n_days: int, L: float, maintenance: float = 0.25,
                 spread_annual: float = 0.0, action: str = "close") -> dict:
    """Fixed loan with a margin call (see ``leverage.margin_loan``) over each window.

    Returns arrays over starts: final equity, minimum equity (incl. start value 1), number of margin calls,
    first call day (1-based within the window, -1 if none) and wiped (equity <= 0 at some close).
    """
    if 0 <= L <= 1:
        raise ValueError("needs L > 1 or L < 0")
    if action not in ("close", "restore"):
        raise ValueError("action must be 'close' or 'restore'")
    r = np.asarray(r, float)
    rf = np.asarray(rf, float)
    starts = np.asarray(starts, int)
    m = len(starts)
    sp = spread_annual / TRADING_DAYS
    long = L > 1
    if long:
        A, B = np.full(m, L), np.full(m, L - 1.0)      # A = position value V, B = debt D
    else:
        A, B = np.full(m, -L), np.full(m, 1.0 - L)     # A = short liability S, B = cash C
    E = np.ones(m)
    minE = np.ones(m)
    ncalls = np.zeros(m, int)
    first = np.full(m, -1, int)
    active = np.ones(m, bool)
    wiped = np.zeros(m, bool)
    for t in range(n_days):
        rt, ft = r[starts + t], rf[starts + t]
        out = ~active & ~wiped
        E = np.where(out, E * (1.0 + ft), E)
        A = np.where(active, A * (1.0 + rt), A)
        B = np.where(active, B * (1.0 + ft + sp), B) if long else np.where(active, B * (1.0 + ft), B)
        Ea = A - B if long else B - A
        w = active & (Ea <= 0)
        wiped |= w
        active &= ~w
        call = active & (Ea < maintenance * A) & (A > 0)
        ncalls += call
        first = np.where(call & (first < 0), t + 1, first)
        if action == "close":
            E = np.where(active, Ea, E)
            E = np.where(w, 0.0, E)
            active &= ~call
        else:
            E = np.where(active, Ea, np.where(w, 0.0, E))
            if long:
                A, B = np.where(call, L * Ea, A), np.where(call, (L - 1.0) * Ea, B)
            else:
                A, B = np.where(call, -L * Ea, A), np.where(call, (1.0 - L) * Ea, B)
        minE = np.minimum(minE, E)
    return {"final": E, "min": minE, "ncalls": ncalls, "first_call": first, "wiped": wiped}


def annualized_growth(final: np.ndarray, n_days: int) -> np.ndarray:
    """Annualized geometric growth of final equity values over n_days (-1 where final <= 0)."""
    years = n_days / TRADING_DAYS
    f = np.asarray(final, float)
    return np.where(f > 0, np.power(np.maximum(f, 1e-300), 1.0 / years) - 1.0, -1.0)
