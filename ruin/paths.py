"""Path statistics on real return series: streaks, drawdowns, adverse excursions, extreme days."""
from __future__ import annotations

import numpy as np
from scipy import stats


def run_lengths(signs: np.ndarray) -> np.ndarray:
    """Lengths of maximal runs of True in a boolean array."""
    x = np.asarray(signs, bool).astype(int)
    if x.size == 0:
        return np.array([], int)
    d = np.diff(np.concatenate([[0], x, [0]]))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return ends - starts


def streak_frequency(r: np.ndarray, kmax: int = 15) -> dict:
    """Observed number of down-day runs of length >= k against the count expected if days were independent.

    Under independence with down-day probability q, the expected number of runs of length >= k in n days is
    approximately (n - k + 1) * (1 - q) * q**k + q**k (runs starting at day 1), used as the reference.
    """
    r = np.asarray(r, float)
    down = r < 0
    q = down.mean()
    n = len(r)
    runs = run_lengths(down)
    ks = np.arange(1, kmax + 1)
    observed = np.array([(runs >= k).sum() for k in ks])
    expected = np.array([(n - k) * (1 - q) * q**k + q**k for k in ks])
    return {"k": ks, "observed": observed, "expected": expected, "q": q, "longest": int(runs.max(initial=0))}


def max_drawdown(path: np.ndarray) -> float:
    path = np.asarray(path, float)
    peak = np.maximum.accumulate(path)
    return float((path / peak - 1.0).min())


def short_adverse_excursion(prices: np.ndarray, start: int, payoff_level: float | None = None) -> dict:
    """For a short sale entered at ``prices[start]``: the largest rise against it before the price first falls
    to ``payoff_level`` (default: the entry price itself, i.e. the moment the trade is no longer under water
    after having been under water). Returns the excursion (as a fraction of the entry price), its day and the
    payoff day (None if never reached)."""
    p = np.asarray(prices, float)
    p0 = p[start]
    level = p0 if payoff_level is None else payoff_level
    future = p[start + 1:]
    hit = np.flatnonzero(future <= level)
    end = hit[0] + 1 if hit.size else len(future)
    window = p[start: start + end + 1]
    j = int(np.argmax(window))
    return {"excursion": float(window[j] / p0 - 1.0), "peak_day": start + j,
            "payoff_day": start + end if hit.size else None}


def extreme_day_counts(r: np.ndarray, thresholds=(3, 4, 5, 6, 8, 10)) -> list[dict]:
    """Days beyond k standard deviations (full-sample mean and sd) against the normal-model expectation."""
    r = np.asarray(r, float)
    z = (r - r.mean()) / r.std(ddof=1)
    n = len(r)
    rows = []
    for k in thresholds:
        rows.append({"k": k, "observed_down": int((z <= -k).sum()), "observed_up": int((z >= k).sum()),
                     "expected_each_side": float(n * stats.norm.sf(k))})
    return rows


def longest_run(signs: np.ndarray) -> tuple[int, int]:
    """(length, start index) of the first longest run of True; (0, -1) if there is none."""
    x = np.asarray(signs, bool).astype(int)
    if x.size == 0 or not x.any():
        return 0, -1
    d = np.diff(np.concatenate([[0], x, [0]]))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    j = int(np.argmax(ends - starts))
    return int(ends[j] - starts[j]), int(starts[j])


def short_trade_outcomes(prices: np.ndarray, starts, horizon: int, min_adverse: float = 0.0) -> dict:
    """Short sales entered at the close ``prices[s]`` for each start s, watched for ``horizon`` days.

    Payoff = the first day the price is back at or below the entry price AFTER having traded more than
    ``min_adverse`` (a fraction) above it (a trade that is never under water has no 'too early' phase). Returns arrays over the starts:
    under_water (price rose above entry at some day in the window), payoff_day (days after entry, -1 if none
    within the window or never under water) and mae (max rise vs entry, up to the payoff day or the window end).
    Only starts with s + horizon < len(prices) should be passed."""
    p = np.asarray(prices, float)
    starts = np.asarray(starts, int)
    m = len(starts)
    under = np.zeros(m, bool)
    payoff = np.full(m, -1, int)
    mae = np.zeros(m)
    for i, s in enumerate(starts):
        w = p[s: s + horizon + 1] / p[s]
        above = np.flatnonzero(w[1:] > 1.0 + min_adverse)
        if above.size == 0:
            mae[i] = max(0.0, w.max() - 1.0)
            continue
        under[i] = True
        t1 = above[0] + 1
        back = np.flatnonzero(w[t1:] <= 1.0)
        end = t1 + back[0] if back.size else len(w) - 1
        if back.size:
            payoff[i] = end
        mae[i] = w[: end + 1].max() - 1.0
    return {"under_water": under, "payoff_day": payoff, "mae": mae}
