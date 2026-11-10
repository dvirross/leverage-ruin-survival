# Leverage, ruin and survival

Reproducible analysis for the Quant Research note
[**"When Does a Strategy That Works Also Survive?"**](https://dvirross.com/quant-research/notes/when-does-a-strategy-that-works-also-survive)
on dvirross.com.

## Run

```sh
pip install -r requirements.txt
python scripts/download_data.py      # Kenneth R. French daily files into data/raw/ (checksums recorded)
python scripts/run_analysis.py       # tables in results/, headline numbers in results/SUMMARY.md (< 1 minute)
python scripts/make_figures.py       # Plotly specs and static images (static export needs kaleido and Chromium)
python -m pytest -q                  # unit tests
```

Data: [Kenneth R. French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html),
daily US market and risk-free returns from July 1926 and Japan market returns (US dollars) from July 1990. The raw
files are downloaded by the script and not redistributed here. Code: MIT licence.

## Question

On real daily US market returns since 1926, how do three quantities change with leverage: the mean return
(the strategy "works"), the median long-run growth, and the probability of being wiped out or forced out by a
margin call within a horizon (the strategy "survives")? Two kinds of leverage are compared: a position reset to
leverage L every day (as in a leveraged ETF) and a fixed loan with a margin call (forced closure or a forced cut back
to the original leverage). Short positions (negative L) model a correct forecast made too early.

## Layout

- `ruin/games.py`: gambler's ruin formula and the doubling (martingale) system; synthetic by design, because the
  rules of the game are the object of study.
- `ruin/leverage.py`: daily-rebalanced leverage with financing at the risk-free rate plus a spread; fixed loan with
  maintenance margin, margin calls, forced closure or restore; Kelly leverage in the lognormal model.
- `ruin/paths.py`: down-day streaks against an independent coin, drawdowns, the adverse excursion of a short sale,
  extreme days against a normal model.
- `ruin/data.py`, `scripts/download_data.py`: the Kenneth R. French Data Library daily files (US market and
  risk-free rate from July 1926; Japan from 1990, in US dollars). Raw files are downloaded, not redistributed.
- `tests/`: `python -m pytest`.

## Conventions

Simple daily returns; 252 trading days a year; only daily closes are observed, so a wipe-out or a margin call is
detected at a close and losses inside a day (gap risk) are not modelled; financing at the daily risk-free rate plus a
stated spread; no taxes or trading costs unless stated.
