"""Leverage on a return series: daily rebalancing, and a fixed loan with a margin call.

Conventions
- ``r`` are simple daily returns of the asset (total return), ``rf`` daily risk-free returns (same length).
- Financing: borrowed money costs rf + spread per day (spread given as an annual rate / 252);
  for L < 1 (including short positions, L < 0) cash earns rf and short proceeds earn rf (no borrow fee unless spread).
- Only daily closes are observed, so a wipe-out can only be detected at a close (gap risk inside the day is not modelled).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TRADING_DAYS = 252


def daily_rebalanced(r: np.ndarray, rf: np.ndarray, L: float, spread_annual: float = 0.0) -> np.ndarray:
    """Equity path (starting at 1) of a position reset to leverage L every day.

    Daily equity return: L*r - (L-1)*(rf + s), with s the daily spread charged on borrowing when L > 1.
    Equity is absorbed at 0: once a day's return is <= -1 the position is wiped out for good.
    """
    r = np.asarray(r, float)
    rf = np.asarray(rf, float)
    s = spread_annual / TRADING_DAYS if L > 1 else 0.0
    eq_ret = L * r - (L - 1.0) * (rf + s)
    growth = np.maximum(1.0 + eq_ret, 0.0)
    path = np.concatenate([[1.0], np.cumprod(growth)])
    return path


@dataclass
class MarginResult:
    equity: np.ndarray          # investor equity at each close (start 1)
    exposure: np.ndarray        # market value of the position at each close
    calls: list[int]            # day indices (1-based in the path) where a margin call was triggered
    closed_at: int | None       # day the position was closed (action='close'), else None
    wiped_out_at: int | None    # first day equity <= 0, else None


def margin_loan(r: np.ndarray, rf: np.ndarray, L: float, maintenance: float = 0.25,
                spread_annual: float = 0.0, action: str = "close") -> MarginResult:
    """Buy-and-hold with a fixed loan: borrow (L-1) per unit of equity once, no daily rebalancing.

    For L > 1 (long with a loan): position value V, debt D grows at rf + spread, equity E = V - D.
    For L < 0 (short sale of |L| units per unit of equity): liability S follows the asset, cash C = 1 + |L|
    grows at rf, equity E = C - S. With total-return data the short is a short of an accumulating (total-return)
    instrument: dividends owed are treated as reinvested in the liability rather than paid from cash, which slightly
    overstates the liability (and makes margin calls slightly earlier) compared with a short of ordinary shares.
    The cash, including the short-sale proceeds, earns the risk-free rate (an institutional rebate; retail
    accounts often earn less). A margin call is triggered when E / gross exposure < ``maintenance``.
    action='close': the position is closed at that close (the investor is forced out and keeps E).
    action='restore': the position is cut back to the original leverage L at that close (losses are locked in).
    """
    if 0 <= L <= 1:
        raise ValueError("margin_loan needs L > 1 (long with a loan) or L < 0 (short)")
    r = np.asarray(r, float)
    rf = np.asarray(rf, float)
    n = len(r)
    s = spread_annual / TRADING_DAYS
    eq = np.empty(n + 1)
    ex = np.empty(n + 1)
    eq[0], ex[0] = 1.0, abs(L)
    calls: list[int] = []
    closed_at = wiped = None
    if L > 1:
        V, D = L, L - 1.0
    else:
        S, C = -L, 1.0 - L  # short |L|; cash = equity + proceeds
    E = 1.0
    for t in range(n):
        if closed_at is not None or wiped is not None:
            # out of the market: equity (if any) earns the risk-free rate
            E = E * (1.0 + rf[t]) if E > 0 else 0.0
            eq[t + 1], ex[t + 1] = E, 0.0
            continue
        if L > 1:
            V *= 1.0 + r[t]
            D *= 1.0 + rf[t] + s
            E, gross = V - D, V
        else:
            S *= 1.0 + r[t]
            C *= 1.0 + rf[t]
            E, gross = C - S, S
        if E <= 0:
            wiped = t + 1
            E = 0.0
            eq[t + 1], ex[t + 1] = 0.0, 0.0
            continue
        if gross > 0 and E / gross < maintenance:
            calls.append(t + 1)
            if action == "close":
                closed_at = t + 1
                eq[t + 1], ex[t + 1] = E, 0.0
                continue
            if action == "restore":
                if L > 1:
                    V, D = L * E, (L - 1.0) * E
                    gross = V
                else:
                    S, C = -L * E, (1.0 - L) * E
                    gross = S
            else:
                raise ValueError("action must be 'close' or 'restore'")
        eq[t + 1], ex[t + 1] = E, gross
    return MarginResult(eq, ex, calls, closed_at, wiped)


def growth_rate(path: np.ndarray) -> float:
    """Annualized geometric growth rate of an equity path (-1 if it was wiped out)."""
    if path[-1] <= 0:
        return -1.0
    years = (len(path) - 1) / TRADING_DAYS
    return float(path[-1] ** (1.0 / years) - 1.0)


def kelly_leverage(mu_excess: float, sigma: float) -> float:
    """Growth-optimal constant leverage in the continuous-time lognormal model: (mu - r) / sigma**2 (annual inputs)."""
    return mu_excess / sigma**2


def approx_growth(L: float, mu: float, r: float, sigma: float) -> float:
    """Continuous-time growth rate of constant leverage L: r + L(mu - r) - L**2 sigma**2 / 2 (annual, log scale)."""
    return r + L * (mu - r) - 0.5 * L**2 * sigma**2
