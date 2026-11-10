"""Betting games whose model is the object of study: gambler's ruin and the doubling (martingale) system.

Synthetic randomness is appropriate here: the point is what the rules of the game imply, not what markets did.
"""
from __future__ import annotations

import numpy as np


def gamblers_ruin_probability(capital: int, target: int, p: float) -> float:
    """Probability of losing everything before reaching ``target``.

    Unit bets, win probability ``p`` per bet, starting with ``capital`` units, stopping at 0 or ``target``.
    Classical result (Feller, ch. XIV): with q = 1 - p and r = q/p,
    P(ruin) = (r**capital - r**target) / (1 - r**target) for p != 1/2, and 1 - capital/target for p = 1/2.
    """
    if not 0 <= capital <= target:
        raise ValueError("need 0 <= capital <= target")
    if not 0 < p < 1:
        raise ValueError("need 0 < p < 1")
    if capital == 0:
        return 1.0
    if capital == target:
        return 0.0
    q = 1.0 - p
    if np.isclose(p, 0.5):
        return 1.0 - capital / target
    r = q / p
    # work in logs when r**target overflows
    if r > 1:
        # divide numerator and denominator by r**target
        num = r ** (capital - target) - 1.0
        den = r ** (-target) - 1.0
        return float(num / den)
    return float((r**capital - r**target) / (1.0 - r**target))


def doubling_cycle_outcomes(base_bet: float, bankroll: float, p: float) -> dict:
    """Exact distribution of one cycle of the doubling system.

    A cycle bets base_bet, doubles after every loss and stops at the first win (net gain base_bet)
    or when the next doubled bet can no longer be covered by the bankroll (net loss of everything staked so far).
    Returns the number of affordable bets k, P(win cycle) = 1 - q**k, the loss when all k bets lose,
    and the expected net result of the cycle, which is 0 for a fair game (p = 1/2), as optional stopping implies.
    """
    if base_bet <= 0 or bankroll < base_bet:
        raise ValueError("bankroll must cover at least one base bet")
    q = 1.0 - p
    k = int(np.floor(np.log2(bankroll / base_bet + 1.0) + 1e-12))  # base*(2**k - 1) <= bankroll
    loss = base_bet * (2**k - 1)
    p_win = 1.0 - q**k
    expected = p_win * base_bet - (q**k) * loss
    return {"bets_affordable": k, "p_win": p_win, "p_lose_all_bets": q**k, "loss_if_lost": loss, "expected": expected}


def simulate_doubling(base_bet: float, bankroll: float, p: float, n_rounds: int, n_players: int,
                      rng: np.random.Generator) -> np.ndarray:
    """Bankroll paths of players who repeat doubling cycles for ``n_rounds`` individual bets.

    After a lost cycle the player restarts with the base bet if the bankroll still covers it; a player
    whose bankroll falls below the base bet is out (bankroll frozen). Returns an array (n_players, n_rounds + 1).
    """
    wealth = np.full(n_players, float(bankroll))
    bet = np.full(n_players, float(base_bet))
    out = np.empty((n_players, n_rounds + 1))
    out[:, 0] = wealth
    for t in range(1, n_rounds + 1):
        alive = wealth >= base_bet
        bet = np.where(bet > wealth, base_bet, bet)  # cannot cover the doubled bet: cycle lost, restart
        alive &= bet <= wealth
        win = rng.random(n_players) < p
        delta = np.where(win, bet, -bet) * alive
        wealth = wealth + delta
        bet = np.where(alive & ~win, bet * 2.0, base_bet)
        out[:, t] = wealth
    return out


def simulate_doubling_at(base_bet: float, bankroll: float, p: float, record_at, n_players: int,
                         rng: np.random.Generator) -> dict:
    """Same dynamics as ``simulate_doubling`` but only the bankrolls after the bet counts in ``record_at``
    are kept (memory-light for long horizons). Returns {n_bets: array of n_players bankrolls}.
    With the same generator state it draws the same random numbers as ``simulate_doubling``."""
    record = sorted(set(int(x) for x in record_at))
    wealth = np.full(n_players, float(bankroll))
    bet = np.full(n_players, float(base_bet))
    out = {}
    for t in range(1, record[-1] + 1):
        alive = wealth >= base_bet
        bet = np.where(bet > wealth, base_bet, bet)
        alive &= bet <= wealth
        win = rng.random(n_players) < p
        wealth = wealth + np.where(win, bet, -bet) * alive
        bet = np.where(alive & ~win, bet * 2.0, base_bet)
        if t in record:
            out[t] = wealth.copy()
    return out
