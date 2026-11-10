import numpy as np
import pytest

from ruin.data import parse_csv_text
from ruin.paths import extreme_day_counts, max_drawdown, run_lengths, short_adverse_excursion, streak_frequency


def test_run_lengths():
    assert run_lengths(np.array([1, 1, 0, 1, 0, 0, 1, 1, 1], bool)).tolist() == [2, 1, 3]


def test_streak_expected_matches_independent_coin():
    rng = np.random.default_rng(3)
    r = rng.choice([-1.0, 1.0], size=200_000)
    out = streak_frequency(r, kmax=6)
    rel = np.abs(out["observed"] - out["expected"]) / out["expected"]
    assert (rel < 0.05).all()


def test_max_drawdown():
    assert max_drawdown(np.array([1, 2, 1, 3, 1.5])) == pytest.approx(-0.5)


def test_short_adverse_excursion():
    p = np.array([100, 120, 150, 130, 99, 80])
    out = short_adverse_excursion(p, 0)
    assert out["excursion"] == pytest.approx(0.5)
    assert out["peak_day"] == 2 and out["payoff_day"] == 4


def test_extreme_days_normal_sample():
    rng = np.random.default_rng(5)
    rows = extreme_day_counts(rng.standard_normal(1_000_000), thresholds=(3,))
    assert rows[0]["observed_down"] == pytest.approx(rows[0]["expected_each_side"], rel=0.1)


SAMPLE = """This file was created by CMPT_ME_BEME_RETS_DAILY using the 202509 CRSP database.
The Tbill return is the simple daily rate.

,Mkt-RF,SMB,HML,RF
19261101,   0.10,  -0.24,  -0.27,   0.013
19261102,   0.45,  -0.33,  -0.06,   0.013

Copyright 2025 Eugene F. Fama and Kenneth R. French
"""


def test_parse_french_daily():
    df = parse_csv_text(SAMPLE)
    assert len(df) == 2
    assert df["MKT"].iloc[0] == pytest.approx(0.0010 + 0.00013)
