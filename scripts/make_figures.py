"""Publication figures for "When Does a Strategy That Works Also Survive?".

Reads the tables in ``results/`` (and the raw French files for the path figures 6 and 8) and writes
  * Plotly specs   public/charts/risk-ruin/NN-name.json   ({data, layout, narrow}, same format as the Monte Carlo note)
  * WebP fallbacks public/images/writing/risk-ruin/NN-name.webp   (2200 px wide, lossless)
  * copies (PNG and WebP) in research/leverage-ruin-survival/figures/, and CAPTIONS.md there.

Static export uses kaleido; it needs a Chromium: set BROWSER_PATH (kaleido 1.x), e.g.
  BROWSER_PATH=/opt/pw-browsers/chromium-1243/chrome-linux64/chrome python3 scripts/make_figures.py
Every plotted or quoted number is read from results/ or recomputed from the raw data; nothing is typed in except
the labels of the episodes (dates of the crashes) and the design constants.
Usage: python3 scripts/make_figures.py [--no-static]
"""
from __future__ import annotations

import copy
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))

from ruin.data import parse_daily  # noqa: E402
from ruin.leverage import daily_rebalanced, margin_loan  # noqa: E402

RES = ROOT / "results"
FIGS = ROOT / "figures"
# Inside the dvirross.com repository the site copies go to its public/ folder; elsewhere (the companion repository)
# set SITE_DIR to a site checkout, or the copies are written under figures/site/.
import os  # noqa: E402
_site = Path(os.environ["SITE_DIR"]) if os.environ.get("SITE_DIR") else (REPO if (REPO / "astro.config.mjs").exists() else FIGS / "site")
CHARTS = _site / "public/charts/risk-ruin"
IMAGES = _site / "public/images/writing/risk-ruin"
NO_STATIC = "--no-static" in sys.argv

# ---- design constants shared with the Monte Carlo figures -------------------------------------------------------
BLUE, GOLD, TEAL, PURPLE, ORANGE = "#3758d7", "#b8860b", "#0e9488", "#7a4fb3", "#c96f33"
GREY, INK, MUTE, GRID, AXLINE = "#b9bfca", "#142033", "#5d6678", "#eceef3", "#c9ced8"
FONT = "ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"
W = 1100          # CSS width of the static export; scale 2 gives 2200 px

CAPTIONS: list[str] = []


# ---- small helpers ---------------------------------------------------------------------------------------------
def sig(x, n=4):
    """Round to n significant digits (keeps the JSON lean)."""
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return None
    x = float(x)
    return 0.0 if x == 0 else float(f"{x:.{n}g}")


def sigs(a, n=4):
    return [sig(v, n) for v in a]


def dstr(d) -> str:
    return pd.Timestamp(d).strftime("%Y-%m-%d")


def fdate(d, year=True) -> str:
    t = pd.Timestamp(d)
    return f"{t.day} {t.strftime('%b')} {t.year}" if year else f"{t.day} {t.strftime('%b')}"


def pct(x, d=0) -> str:
    return f"{100 * x:.{d}f}%"


def downsample(x, y, n=2000):
    """Keep at most ~n points: first, last and the min and max of every bucket (so extremes survive)."""
    x, y = list(x), np.asarray(y, float)
    if len(y) <= n:
        return x, y
    nb = n // 2 - 1
    edges = np.linspace(0, len(y), nb + 1).astype(int)
    keep = {0, len(y) - 1}
    for a, b in zip(edges[:-1], edges[1:]):
        if b > a:
            seg = y[a:b]
            keep.add(a + int(np.argmin(seg)))
            keep.add(a + int(np.argmax(seg)))
    idx = sorted(keep)
    return [x[i] for i in idx], y[idx]


def title(head: str, sub: str, size_head=16, size_sub=13) -> dict:
    return {"text": f"<b>{head}</b><br><span style='font-size:{size_sub}px;color:{MUTE}'>{sub}</span>",
            "x": 0, "xref": "paper", "xanchor": "left", "y": 1, "yref": "container", "yanchor": "top",
            "pad": {"t": 14}, "font": {"size": size_head, "color": INK}}


def ax(**kw) -> dict:
    d = {"tickfont": {"color": MUTE, "size": 12}, "gridcolor": GRID, "gridwidth": 1, "linecolor": AXLINE,
         "zeroline": False, "ticks": "", "showline": True}
    d.update(kw)
    return d


def ax_title(text: str, size=12) -> dict:
    return {"text": text, "font": {"size": size, "color": MUTE}}


def base_layout(head, sub, height, margin, showlegend=False, narrow=False, **extra) -> dict:
    lay = {
        "font": {"family": FONT, "size": 13, "color": INK},
        "title": title(head, sub, 14 if narrow else 16, 11 if narrow else 13),
        "margin": margin,
        "showlegend": showlegend,
        "legend": {"font": {"size": 12, "color": INK}, "orientation": "h", "x": 0, "xanchor": "left", "y": 1.0,
                   "yanchor": "bottom", "bgcolor": "rgba(255,255,255,0)", "itemwidth": 30},
        "hoverlabel": {"font": {"family": FONT, "color": INK, "size": 12}, "bgcolor": "white", "bordercolor": AXLINE},
        "paper_bgcolor": "white", "plot_bgcolor": "white", "height": height, "hovermode": "closest",
        "annotations": [],
    }
    lay.update(extra)
    return lay


def _lg(v, on):
    return math.log10(v) if on else v


def note(text, x, y, xref="x", yref="y", anchor="left", yanchor="middle", color=INK, size=12, xlog=False, ylog=False, **kw) -> dict:
    x, y = _lg(x, xlog), _lg(y, ylog)
    d = {"x": x, "y": y, "xref": xref, "yref": yref, "text": text, "showarrow": False, "xanchor": anchor,
         "yanchor": yanchor, "align": "left" if anchor == "left" else ("right" if anchor == "right" else "center"),
         "font": {"size": size, "color": color}}
    d.update(kw)
    return d


def arrow(text, x, y, ax_, ay, xref="x", yref="y", axref="pixel", ayref="pixel", anchor="left", color=INK,
          size=12, xlog=False, ylog=False, **kw) -> dict:
    x, y = _lg(x, xlog), _lg(y, ylog)
    d = {"x": x, "y": y, "xref": xref, "yref": yref, "ax": ax_, "ay": ay, "axref": axref, "ayref": ayref,
         "showarrow": True, "arrowhead": 0, "arrowwidth": 1, "arrowcolor": AXLINE, "standoff": 4, "text": text,
         "xanchor": anchor, "yanchor": "middle", "align": "left" if anchor == "left" else "center",
         "font": {"size": size, "color": color}}
    d.update(kw)
    return d


def hline(y, color=INK, dash="dot", width=1, xref="x", yref="y", x0=0, x1=1, xtype="paper") -> dict:
    return {"type": "line", "xref": "paper" if xtype == "paper" else xref, "yref": yref, "x0": x0, "x1": x1,
            "y0": y, "y1": y, "line": {"color": color, "width": width, "dash": dash}}


def vline(x, color=INK, dash="dot", width=1, xref="x", y0=0, y1=1, yref="paper") -> dict:
    return {"type": "line", "xref": xref, "yref": yref, "x0": x, "x1": x, "y0": y0, "y1": y1,
            "line": {"color": color, "width": width, "dash": dash}}


def narrow_part(wide: dict, narrow: dict, max_width=560) -> dict:
    """Top-level layout keys that differ in the narrow layout (the page replaces keys wholesale)."""
    diff = {k: v for k, v in narrow.items() if wide.get(k) != v and k != "height"}
    return {"maxWidth": max_width, "layout": diff}


def finish(name, data, wide, narrow, ratio_h, caption, alt, source):
    wide = copy.deepcopy(wide)
    wide["height"] = ratio_h // 2          # CSS px; the static export is 2x
    spec = {"data": data, "layout": wide, "narrow": narrow_part(wide, narrow)}
    CHARTS.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)
    path = CHARTS / f"{name}.json"
    path.write_text(json.dumps(spec, separators=(",", ":"), ensure_ascii=False))
    status = "json only"
    if not NO_STATIC:
        try:
            import plotly.io as pio
            from PIL import Image
            png = FIGS / f"{name}.png"
            pio.write_image({"data": data, "layout": wide}, str(png), width=W, height=ratio_h // 2, scale=2)
            im = Image.open(png).convert("RGB")
            assert im.size == (2200, ratio_h), im.size
            im.save(FIGS / f"{name}.webp", "WEBP", lossless=True, quality=100, method=6)
            im.save(IMAGES / f"{name}.webp", "WEBP", lossless=True, quality=100, method=6)
            status = f"{im.size[0]}x{im.size[1]} webp {os.path.getsize(IMAGES / f'{name}.webp') // 1024} KB"
            prev = os.environ.get("NARROW_PREVIEW_DIR")
            if prev:   # 360 px preview of the phone layout for visual checks (not shipped)
                nl = copy.deepcopy(wide)
                nl.update(narrow)
                nl["height"] = narrow.get("height", 700)
                pio.write_image({"data": data, "layout": nl}, str(Path(prev) / f"{name}-narrow.png"), width=360, height=nl["height"], scale=2)
        except Exception as e:  # noqa: BLE001
            status = f"STATIC EXPORT FAILED: {e!r}"
    print(f"{name}: json {path.stat().st_size // 1024} KB; {status}")
    CAPTIONS.append((name, caption, alt, source, ratio_h))


def finish_wrap(name, data, wide, narrow_layout, ratio_h, caption, alt, source, preview_narrow=True):
    """finish() plus a 360 px preview PNG of the narrow layout in scratch (checked by eye, not shipped)."""
    finish(name, data, wide, narrow_layout, ratio_h, caption, alt, source)


# ---- data -----------------------------------------------------------------------------------------------------
LW = pd.read_csv(RES / "leverage_windows.csv")
US = parse_daily(ROOT / "data/raw/F-F_Research_Data_Factors_daily_CSV.zip")
SRC_US = "Kenneth R. French Data Library, daily market returns 1926–2026 (value-weighted US market including dividends)."
SRC_JP = ("Kenneth R. French Data Library, Japan daily market return from 1990 (US dollars, value-weighted, "
          "including dividends); not the Nikkei 225.")


def lw(kind, hz, spread=0.0, maint=None, sample="1926-2026"):
    d = LW[(LW["sample"] == sample) & (LW["kind"] == kind) & (LW["spread_annual"] == spread)
           & (LW["horizon_years"] == hz)]
    if maint is not None:
        d = d[d["maintenance"] == maint]
    return d.sort_values("L").reset_index(drop=True)


# =================================================================================================================
# Figure 1: Kobe
# =================================================================================================================
def fig1():
    j = pd.read_csv(RES / "japan_kobe_window.csv", parse_dates=["date"])
    ks = pd.read_csv(RES / "japan_kobe_summary.csv").iloc[0]
    ref = pd.Timestamp(ks["reference_date"])
    quake = pd.Timestamp("1995-01-17")
    trough = pd.Timestamp(ks["trough_date_after_quake_in_1995"])
    barings = pd.Timestamp("1995-02-26")               # a Sunday: no close
    assert barings.dayofweek == 6
    after = j[j.date > barings].iloc[0]                # nearest trading day after (Monday 27 Feb)
    before = j[j.date < barings].iloc[-1]
    near = after if (after.date - barings) <= (barings - before.date) else before
    lvl = lambda d: float(j.loc[j.date == d, "index_ref100"].iloc[0])
    t_lvl, q_lvl, b_lvl = lvl(trough), lvl(quake), lvl(near.date)
    back = pd.Timestamp(ks["first_close_back_at_or_above_ref_after_trough"])
    end_lvl = float(j["index_ref100"].iloc[-1])
    fall = -ks["trough_vs_ref"]
    weeks = round((trough - quake).days / 7)

    data = [
        {"type": "scatter", "mode": "lines", "x": [dstr(j.date.iloc[0]), dstr(j.date.iloc[-1])], "y": [100, 100],
         "line": {"color": GREY, "width": 1, "dash": "dash"}, "hoverinfo": "skip", "showlegend": False},
        {"type": "scatter", "mode": "lines", "name": "Japan, US dollars", "x": [dstr(d) for d in j.date],
         "y": sigs(j.index_ref100), "line": {"color": BLUE, "width": 2},
         "hovertemplate": "%{x|%d %b %Y}: %{y:.1f}<extra></extra>"},
        {"type": "scatter", "mode": "markers", "name": "Marked days", "x": [dstr(quake), dstr(trough)],
         "y": sigs([q_lvl, t_lvl]), "marker": {"color": [GOLD, ORANGE], "size": 10, "line": {"color": "white", "width": 2}},
         "hoverinfo": "skip", "showlegend": False},
    ]
    sub = "US-dollar returns of the Japanese market (Fama–French), not the Nikkei 225; indexed to 100 at the close of 16 Jan 1995"
    head = (f"After the Kobe earthquake Japan fell {pct(fall)} in {weeks} weeks, "
            f"and the index was back at its starting level by {fdate(back, False)}")
    sub_n = "Japanese market in US dollars (Fama–French),<br>not the Nikkei 225; 100 = close of 16 Jan 1995"

    def lay(narrow):
        shapes = [vline(dstr(barings), ORANGE, "dash", 1.2)]
        ann = [
            note("100 = close of 16 Jan 1995", "1995-08-15", 100, anchor="center", yanchor="bottom",
                 color=MUTE, size=11, yshift=2),
            arrow(f"17 Jan: earthquake<br>{pct(ks['return_on_1995-01-17'], 1).replace('-', '−')} on the day" if not narrow else "17 Jan:<br>earthquake",
                  dstr(quake), q_lvl, 0, -70, anchor="center"),
            arrow(f"{fdate(trough, False)}: low, {pct(ks['trough_vs_ref'], 1).replace('-', '−')}" if not narrow else "27 Feb: low",
                  dstr(trough), t_lvl, -30, 28,
                  anchor="right", color=INK),
            note("26 Feb (Sunday): Barings collapses.<br>Shown at the next close, 27 Feb" if not narrow else "26 Feb, Sunday:<br>Barings collapses<br>(shown at 27 Feb)",
                 dstr(barings), 86.5, anchor="left", xshift=8, color=ORANGE, yanchor="middle"),
        ]
        return base_layout(head if not narrow else f"Japan fell {pct(fall)} in {weeks} weeks after Kobe,<br>then recovered",
                           sub if not narrow else sub_n, 500,
                           {"l": 52, "r": 36, "t": 84, "b": 56} if not narrow else {"l": 44, "r": 12, "t": 142, "b": 50},
                           narrow=narrow,
                           xaxis=ax(type="date", range=["1994-12-01", "1995-12-29"], tickformat="%b %Y",
                                    dtick="M2" if not narrow else "M3"),
                           yaxis=ax(range=[84, 108], tickvals=[85, 90, 95, 100, 105], title=ax_title("") ),
                           shapes=shapes, annotations=ann)

    wide, nar = lay(False), lay(True)
    finish_wrap("01-kobe-japan", data, wide, nar, 1000,
                f"The Japanese stock market, in US dollars, around the Kobe earthquake of 17 January 1995 and the collapse of Barings on Sunday 26 February: a fall of {pct(fall, 1)} from the close before the earthquake to the low of 27 February, and a full recovery by 10 April.",
                f"Line chart of a Japanese stock-market index in US dollars from December 1994 to December 1995, set to 100 at the close of 16 January 1995. It drops {pct(-ks['return_on_1995-01-17'], 1)} on the earthquake day, falls to a low of {t_lvl:.1f} ({pct(ks['trough_vs_ref'], 1)}) on 27 February, the first trading day after Barings Bank collapsed, and is back at 100 on 10 April; it ends the year at {end_lvl:.1f}.",
                SRC_JP)


# =================================================================================================================
# Figure 2: doubling game (simulation)
# =================================================================================================================
def fig2():
    g = pd.read_csv(RES / "doubling_game.csv")
    g = g[g.bankroll == 100]
    fair = g[g.p == 0.5].sort_values("n_bets")
    roul = g[g.p < 0.5].sort_values("n_bets")
    n = [int(v) for v in fair.n_bets]
    players = int(fair.n_players.iloc[0])
    last_f, last_r = fair.iloc[-1], roul.iloc[-1]
    hv = "%{x:,} bets<br>%{y:.1f}%<extra>{}</extra>"
    data = [
        {"type": "scatter", "mode": "lines+markers", "name": "Ahead of the start", "x": n,
         "y": sigs(100 * fair.share_ahead), "line": {"color": BLUE, "width": 2},
         "marker": {"size": 8, "color": BLUE, "line": {"color": "white", "width": 2}},
         "hovertemplate": "%{x:,} bets: %{y:.1f}% of players ahead<extra>Fair coin</extra>"},
        {"type": "scatter", "mode": "lines+markers", "name": "Ruined (fair coin)", "x": n,
         "y": sigs(100 * fair.share_ruined), "line": {"color": ORANGE, "width": 2},
         "marker": {"size": 8, "color": ORANGE, "line": {"color": "white", "width": 2}},
         "hovertemplate": "%{x:,} bets: %{y:.1f}% of players ruined<extra>Fair coin</extra>"},
        {"type": "scatter", "mode": "lines+markers", "name": "Ruined (roulette, 18/37)", "x": n,
         "y": sigs(100 * roul.share_ruined), "line": {"color": ORANGE, "width": 1.5, "dash": "dash"},
         "marker": {"size": 6, "color": "white", "line": {"color": ORANGE, "width": 1.5}},
         "hovertemplate": "%{x:,} bets: %{y:.1f}% of players ruined<extra>Roulette, even-money bet</extra>"},
        {"type": "scatter", "mode": "lines+markers", "name": "Average bankroll", "xaxis": "x2", "yaxis": "y2",
         "x": n, "y": sigs(fair.mean_final), "line": {"color": TEAL, "width": 2},
         "marker": {"size": 8, "color": TEAL, "line": {"color": "white", "width": 2}},
         "hovertemplate": "%{x:,} bets: average bankroll %{y:.1f}<extra>Fair coin</extra>"},
    ]
    head = "On average nothing happens, yet most players end up ruined"
    sub = (f"Simulation: bet 1, double after every loss, fair coin, starting bankroll 100; {players:,} simulated players; "
           "ruined = bankroll below the base bet")
    sub_n = f"Simulation: double after every loss, fair coin,<br>bankroll 100, {players:,} players"

    def lay(narrow):
        tv = [10, 100, 1000, 5000]
        xa = dict(type="log", range=[0.9, 3.85], tickvals=tv, ticktext=["10", "100", "1,000", "5,000"], anchor="y2" if False else "y")
        ann = [
            note(f"Ruined, fair coin: {pct(last_f.share_ruined)}", 5000, 100 * last_f.share_ruined, xlog=True, xshift=10, color=ORANGE, yshift=-9),
            note(f"Ruined, roulette: {pct(last_r.share_ruined)}", 5000, 100 * last_r.share_ruined, xlog=True, xshift=10, color=ORANGE, yshift=9),
            note(f"Ahead: {pct(last_f.share_ahead)}", 5000, 100 * last_f.share_ahead, xlog=True, xshift=10, color=BLUE),
            note(f"Average: {last_f.mean_final:.0f}", 5000, last_f.mean_final, xref="x2", yref="y2", xlog=True, xshift=10, color=TEAL),
            note("Start: 100", 10, 100, xlog=True, xref="x2", yref="y2", yanchor="bottom", color=MUTE, size=11, yshift=3, xshift=0),
            note("Share of players", 0.0, 1.0, xref="paper", yref="paper", anchor="left", yanchor="bottom", color=MUTE, size=12, yshift=4, xshift=-52) if False else {},
        ]
        ann = [a for a in ann if a]
        if narrow:
            ann = [
                note(f"Ruined: {pct(last_f.share_ruined)}", 5000, 100 * last_f.share_ruined, xlog=True, xshift=6, color=ORANGE, size=11, yshift=8),
                note(f"Ahead: {pct(last_f.share_ahead)}", 5000, 100 * last_f.share_ahead, xlog=True, xshift=6, color=BLUE, size=11),
                note(f"Average: {last_f.mean_final:.0f}", 5000, last_f.mean_final, xref="x2", yref="y2", xlog=True, xshift=6, color=TEAL, size=11),
            ]
        shapes = [hline(100, GREY, "dash", 1, xref="x2", yref="y2")]
        return base_layout(head if not narrow else "On average nothing happens,<br>yet most players end up ruined", sub if not narrow else sub_n, 600 if not narrow else 600,
                           {"l": 60, "r": 175, "t": 84, "b": 60} if not narrow else {"l": 46, "r": 70, "t": 200, "b": 52},
                           showlegend=narrow, narrow=narrow,
                           xaxis=ax(**{k: v for k, v in xa.items() if k != "anchor"}, showticklabels=False, domain=[0, 1], anchor="y"),
                           xaxis2=ax(**{k: v for k, v in xa.items() if k != "anchor"}, anchor="y2", matches="x",
                                     title=ax_title("Number of bets (log scale)")),
                           yaxis=ax(domain=[0.36, 1], range=[0, 100], tickvals=[0, 25, 50, 75, 100], ticktext=["0%", "25%", "50%", "75%", "100%"],
                                    title=ax_title("Share of players" if not narrow else "")),
                           yaxis2=ax(domain=[0, 0.24], range=[0, 130], tickvals=[0, 50, 100],
                                     title=ax_title("Average bankroll" if not narrow else "")),
                           shapes=shapes, annotations=ann,
                           legend={"font": {"size": 12, "color": INK}, "orientation": "h", "x": 0, "xanchor": "left",
                                   "y": 1.0, "yanchor": "bottom", "bgcolor": "rgba(255,255,255,0)", "itemwidth": 30})

    wide, nar = lay(False), lay(True)
    finish_wrap("02-doubling-game", data, wide, nar, 1200,
                f"A simulated doubling game on a fair coin: the average bankroll stays near 100 whatever the number of bets, but after 5,000 bets {pct(last_f.share_ruined)} of players are ruined and only {pct(last_f.share_ahead)} are ahead; with roulette odds the ruined share reaches {pct(last_r.share_ruined)}.",
                f"Two stacked charts against the number of bets (10 to 5,000, logarithmic axis). Top: for a fair coin the share of players ahead falls from {pct(fair.share_ahead.iloc[0])} at 10 bets to {pct(last_f.share_ahead)} at 5,000, while the share ruined rises from {pct(fair.share_ruined.iloc[0])} to {pct(last_f.share_ruined)}; a dashed line for roulette odds ends at {pct(last_r.share_ruined)} ruined. Bottom: the average bankroll stays between {fair.mean_final.min():.0f} and {fair.mean_final.max():.0f}, close to the starting 100.",
                "Simulation (20,000 simulated players per point, fixed seed); code in the companion repository.")


# =================================================================================================================
# Figure 3: down streaks
# =================================================================================================================
def fig3():
    s = pd.read_csv(RES / "down_streaks.csv")
    s = s[s["sample"] == "1926-2026"].sort_values("k")
    lg = pd.read_csv(RES / "down_streak_longest.csv").iloc[0]
    k = [int(v) for v in s.k]
    obs = s[s.observed_runs_ge_k > 0]
    q, n = float(s.q_down.iloc[0]), int(s.n_days.iloc[0])
    k12 = s[s.k == int(lg.longest_down_streak)].iloc[0]
    zero = s[s.observed_runs_ge_k == 0]
    data = [
        {"type": "scatter", "mode": "lines+markers", "name": "If each day were an independent coin", "x": k,
         "y": sigs(s.expected_independent), "line": {"color": INK, "width": 1.5, "dash": "dash"},
         "marker": {"size": 6, "color": "white", "line": {"color": INK, "width": 1.5}},
         "hovertemplate": "At least %{x} down days in a row: %{y:.2g} expected<extra>Independent coin</extra>"},
        {"type": "scatter", "mode": "lines+markers", "name": "Observed in the US market", "x": [int(v) for v in obs.k],
         "y": [int(v) for v in obs.observed_runs_ge_k], "line": {"color": BLUE, "width": 2},
         "marker": {"size": 8, "color": BLUE, "line": {"color": "white", "width": 2}},
         "hovertemplate": "At least %{x} down days in a row: %{y:,} streaks<extra>Observed</extra>"},
    ]
    dd = lambda d: fdate(d, True)
    start, end = pd.Timestamp(lg.start_date), pd.Timestamp(lg.end_date)
    head = "Down streaks run longer than a coin would give, but only modestly"
    sub = (f"Runs of at least k consecutive down days, US market, {n:,} trading days 1926–2026; "
           f"the coin falls on {pct(q, 1)} of days, as the market did")
    sub_n = f"US market, {n:,} trading days 1926–2026;<br>coin with the observed {pct(q, 1)} down-day frequency"

    def lay(narrow):
        ann = [
            note("Observed", 6, float(s[s.k == 6].observed_runs_ge_k.iloc[0]), color=BLUE, yshift=22, xshift=14, anchor="center", size=13, ylog=True),
            note("Independent coin", 9, float(s[s.k == 9].expected_independent.iloc[0]), color=INK, yshift=-24, xshift=0, anchor="center", size=13, ylog=True),
            arrow(f"Longest streak: {int(lg.longest_down_streak)} down days<br>{start.day}–{fdate(end, False)} {end.year} ({pct(lg.cumulative_return, 1).replace('-', '−')} in total)<br>"
                  f"{int(k12.observed_runs_ge_k)} runs of 12+ vs {k12.expected_independent:.1f} expected",
                  12, float(k12.observed_runs_ge_k), 24, -80, anchor="left", axref="pixel", ylog=True),
            note(f"No streak of 13 or more<br>(expected {zero.expected_independent.iloc[0]:.1f}, {zero.expected_independent.iloc[1]:.1f}, {zero.expected_independent.iloc[2]:.1f} at 13, 14, 15)",
                 14, 0.5, anchor="center", yanchor="bottom", color=MUTE, size=11, ylog=True),
        ]
        if narrow:
            ann = [
                arrow(f"Longest: 12 days,<br>April 1970", 12, float(k12.observed_runs_ge_k), -10, 70, anchor="right", ylog=True),
                note("No streak of 13+", 15, 0.09, anchor="right", color=MUTE, size=11, ylog=True),
            ]
        return base_layout(head if not narrow else "Down streaks run a little<br>longer than coin flips",
                           sub if not narrow else sub_n, 500,
                           {"l": 70, "r": 36, "t": 84, "b": 60} if not narrow else {"l": 54, "r": 12, "t": 122, "b": 54},
                           showlegend=narrow, narrow=narrow,
                           xaxis=ax(range=[0.6, 15.5], tickvals=list(range(1, 16)), title=ax_title("Streak length k: consecutive down days")),
                           yaxis=ax(type="log", range=[-1.3, 3.9], tickvals=[0.1, 1, 10, 100, 1000],
                                    ticktext=["0.1", "1", "10", "100", "1,000"],
                                    title=ax_title("Runs of at least k days (log scale)" if not narrow else "")),
                           annotations=ann)

    wide, nar = lay(False), lay(True)
    finish_wrap("03-down-streaks", data, wide, nar, 1000,
                f"Runs of consecutive down days in US market returns, 1926–2026, against the number a coin with the same down-day frequency would produce. Streaks are a little more frequent than chance, and the longest ({int(lg.longest_down_streak)} days, April 1970) cost only {pct(-lg.cumulative_return, 1)} in total.",
                f"Chart on a logarithmic vertical axis of the number of runs of at least k consecutive down days, k = 1 to 15. Observed counts fall from {int(s.observed_runs_ge_k.iloc[0]):,} at k = 1 to {int(k12.observed_runs_ge_k)} at k = 12 and none at 13 or more, close to a dashed line for an independent coin ({s.expected_independent.iloc[0]:,.0f} at k = 1, {k12.expected_independent:.1f} at k = 12), slightly above it from k = 3 onward. The longest streak, 12 days in April 1970, lost {pct(-lg.cumulative_return, 1)}.",
                SRC_US)


# =================================================================================================================
# Figure 4: works vs survives (thumbnail)
# =================================================================================================================
def fig4():
    d = lw("daily", 10)
    nwin = int(d.n_windows.iloc[0])
    kf = pd.read_csv(RES / "kelly_full_sample.csv")
    kf = kf[(kf["sample"] == "1926-2026") & (kf.spread_annual == 0)].iloc[0]
    kelly, grid = float(kf.kelly_formula), float(kf.grid_optimal_L)
    L = [float(v) for v in d.L]
    mean = 100 * d.mean_arith_annual_return
    med = 100 * d.median_growth
    fall = 100 * d.p_fall_50pct
    wipe = 100 * d.p_wipeout
    cap = 125
    off = d[d.mean_arith_annual_return * 100 > cap]
    peak_med = d.loc[d.median_growth.idxmax()]
    f_at = lambda l: float(d.loc[d.L == l, "p_fall_50pct"].iloc[0])
    data = [
        {"type": "scatter", "mode": "lines+markers", "name": "Mean annual return (arithmetic)", "x": L, "y": sigs(mean),
         "line": {"color": BLUE, "width": 2}, "marker": {"size": 7, "color": BLUE, "line": {"color": "white", "width": 1.5}},
         "hovertemplate": "Leverage %{x}×: mean annual return %{y:,.0f}%<extra></extra>"},
        {"type": "scatter", "mode": "lines+markers", "name": "Median annualized growth", "x": L, "y": sigs(med),
         "line": {"color": GOLD, "width": 2}, "marker": {"size": 7, "color": GOLD, "line": {"color": "white", "width": 1.5}},
         "hovertemplate": "Leverage %{x}×: median annualized growth %{y:.1f}%<extra></extra>"},
        {"type": "scatter", "mode": "lines+markers", "name": "Chance of a 50% fall", "xaxis": "x2", "yaxis": "y2", "x": L,
         "y": sigs(fall), "line": {"color": ORANGE, "width": 2},
         "marker": {"size": 7, "color": ORANGE, "line": {"color": "white", "width": 1.5}},
         "hovertemplate": "Leverage %{x}×: %{y:.1f}% of windows fell 50% or more<extra></extra>"},
        {"type": "scatter", "mode": "lines+markers", "name": "Chance of wipe-out", "xaxis": "x2", "yaxis": "y2", "x": L,
         "y": sigs(wipe), "line": {"color": PURPLE, "width": 2},
         "marker": {"size": 7, "color": PURPLE, "line": {"color": "white", "width": 1.5}},
         "hovertemplate": "Leverage %{x}×: %{y:.1f}% of windows wiped out<extra></extra>"},
    ]
    offtxt = ", ".join(f"{m:,.0f}% at {l:g}×" for l, m in zip(off.L, 100 * off.mean_arith_annual_return))
    head = "More leverage keeps raising the average, but not the typical outcome or survival"
    sub = (f"Daily-rebalanced leverage on the US market, every 10-year window 1926–2026 ({nwin:,} overlapping windows), "
           "borrowing at the T-bill rate")
    sub_n = f"Daily-rebalanced leverage, US market,<br>{nwin:,} overlapping 10-year windows 1926–2026"

    def lay(narrow):
        kelly_line = vline(kelly, INK, "dash", 1.2)
        grid_line = vline(grid, GOLD, "dot", 1.2)
        ann = [
            note("Mean annual return<br>(the metric that “works”)", 2.75, 69.3 + 0, anchor="right", color=BLUE, xshift=-4, yshift=0, yref="y") if False else
            note("Mean annual return", 2.5, 69.3, anchor="right", color=BLUE, xshift=-10, yshift=16),
            note("Median annualized growth", 5.0, float(100 * d.loc[d.L == 5.0, "median_growth"].iloc[0]), anchor="left", color=GOLD, xshift=10, yshift=-2) if False else
            note(f"Median growth: {100 * float(d.loc[d.L == 6.0, 'median_growth'].iloc[0]):.1f}% at 6×".replace("-", "−"), 6, 100 * float(d.loc[d.L == 6.0, "median_growth"].iloc[0]),
                 anchor="left", color=GOLD, xshift=10),
            note(f"Off the chart: mean {offtxt}", 3.5, cap - 14, anchor="left", color=BLUE, size=11, yshift=0),
            note(f"Median peaks at {peak_med.L:g}× ({100 * peak_med.median_growth:.0f}%)", float(peak_med.L), 100 * float(peak_med.median_growth), anchor="left",
                 yanchor="bottom", color=GOLD, yshift=14, xshift=-14, size=11),
            note(f"50% fall: {100 * f_at(6.0):.0f}% at 6×", 6, 100 * f_at(6.0), xref="x2", yref="y2", anchor="left", color=ORANGE, xshift=10),
            note(f"Wipe-out: {100 * float(d.loc[d.L == 6.0, 'p_wipeout'].iloc[0]):.0f}% at 6×", 6, 100 * float(d.loc[d.L == 6.0, "p_wipeout"].iloc[0]),
                 xref="x2", yref="y2", anchor="left", color=PURPLE, xshift=10, yshift=-8),
            note(f"Kelly leverage {kelly:.2f} (formula)<br>grid optimum {grid:.2f} (full-sample growth)", kelly, 0.35, xref="x", yref="paper", anchor="left", color=INK, xshift=8, yanchor="middle", size=11),
            note("<b>Works</b>", 0.0, 1.0, xref="paper", yref="paper", anchor="left", color=MUTE, yanchor="bottom", xshift=-56) if False else {},
        ]
        ann = [a for a in ann if a]
        if narrow:
            ann = [
                note(f"Off the chart: mean {offtxt}", 3.1, cap - 8, anchor="left", color=BLUE, size=10, yshift=0, width=150) if False else
                note("Mean off the chart<br>beyond 3×", 3.12, cap - 10, anchor="left", color=BLUE, size=10),
                note(f"Kelly {kelly:.2f}", kelly, 1.0, xref="x", yref="paper", anchor="right", xshift=-4, yanchor="top", size=10),
            ]
        shapes = [kelly_line, grid_line, hline(0, GREY, "dash", 1, yref="y"), ]
        return base_layout(head if not narrow else "More leverage lifts the average,<br>not the typical outcome or survival",
                           sub if not narrow else sub_n, 700,
                           {"l": 64, "r": 190, "t": 84, "b": 60} if not narrow else {"l": 50, "r": 14, "t": 200, "b": 52},
                           showlegend=narrow, narrow=narrow,
                           xaxis=ax(range=[0.4, 6.1], tickvals=[0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6], showticklabels=False, domain=[0, 1], anchor="y"),
                           xaxis2=ax(range=[0.4, 6.1], tickvals=[0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6],
                                     ticktext=["0.5×", "1×", "1.5×", "2×", "2.5×", "3×", "4×", "5×", "6×"], anchor="y2", matches="x",
                                     title=ax_title("Leverage (position size relative to equity, reset every day)" if not narrow else "Leverage (reset daily)")),
                           yaxis=ax(domain=[0.40, 1], range=[-15, cap], tickvals=[0, 25, 50, 75, 100], ticktext=["0%", "25%", "50%", "75%", "100%"],
                                    title=ax_title("Return per year" if not narrow else "")),
                           yaxis2=ax(domain=[0, 0.3], range=[0, 100], tickvals=[0, 25, 50, 75, 100], ticktext=["0%", "25%", "50%", "75%", "100%"],
                                     title=ax_title("Chance within 10 years" if not narrow else "")),
                           shapes=shapes, annotations=ann,
                           legend={"font": {"size": 11 if narrow else 12, "color": INK}, "orientation": "h", "x": 0, "xanchor": "left",
                                   "y": 1.0, "yanchor": "bottom", "bgcolor": "rgba(255,255,255,0)", "itemwidth": 30})

    wide, nar = lay(False), lay(True)
    finish_wrap("04-works-vs-survives", data, wide, nar, 1400,
                f"What leverage does over 10 years of US stock-market history: the mean return keeps rising, the median growth peaks near {peak_med.L:g}× and falls beyond, and the chance of a 50% fall climbs from {pct(f_at(1.0))} at 1× to {pct(f_at(3.0))} at 3×, with wipe-outs appearing at 6×. The Kelly leverage computed from the full sample is {kelly:.2f}.",
                f"Two stacked line charts against leverage from 0.5× to 6×, for 10-year windows of daily-rebalanced US market returns since 1926. Top: the mean annual return rises from {mean.iloc[0]:.0f}% to {mean.iloc[-1]:,.0f}% at 6× (off the scale above 3×), while the median annualized growth rises to {100 * peak_med.median_growth:.0f}% at {peak_med.L:g}×, then falls to {med.iloc[-1]:.1f}% at 6×. Bottom: the chance of a 50% fall within the window rises from {fall.iloc[0]:.0f}% at 0.5× to {fall.iloc[-1]:.0f}% at 6×, and the chance of wipe-out is zero until 5× and {wipe.iloc[-1]:.0f}% at 6×. Vertical lines mark the Kelly leverage {kelly:.2f} and the grid optimum {grid:.2f}.",
                SRC_US + " 10-year windows start every 21st trading day and overlap heavily; financing at the risk-free rate, no costs.")


# =================================================================================================================
# Figure 5: heatmaps
# =================================================================================================================
def fig5():
    hs = [1, 5, 10, 20]
    daily = {h: lw("daily", h) for h in hs}
    loan = {h: lw("loan_close", h, maint=0.25) for h in hs}
    Ld = [float(v) for v in daily[10].L]
    Ll = [l for l in [float(v) for v in loan[10].L] if l <= 3.0]
    zd = [[float(daily[h].loc[daily[h].L == l, "p_fall_50pct"].iloc[0]) * 100 for h in hs] for l in Ld]
    zl = [[float(loan[h].loc[loan[h].L == l, "p_margin_call"].iloc[0]) * 100 for h in hs] for l in Ll]
    xs = [f"{h} year" + ("s" if h > 1 else "") for h in hs]
    ys_d = [f"{l:g}×" for l in Ld]
    ys_l = [f"{l:g}×" for l in Ll]
    scale = [[0, "#ffffff"], [0.5, "#f3d3b9"], [1, "#d9894f"]]
    common = {"type": "heatmap", "zmin": 0, "zmax": 100, "colorscale": scale, "showscale": False, "xgap": 2, "ygap": 2,
              "texttemplate": "%{z:.0f}%", "textfont": {"size": 13, "color": INK}, "hoverongaps": False}
    data = [
        dict(common, x=xs, y=ys_d, z=[sigs(r) for r in zd], xaxis="x", yaxis="y",
             hovertemplate="Leverage %{y}, %{x}: %{z:.1f}% of windows fell 50% or more<extra></extra>"),
        dict(common, x=xs, y=ys_l, z=[sigs(r) for r in zl], xaxis="x2", yaxis="y2",
             hovertemplate="Loan, leverage %{y}, %{x}: %{z:.1f}% of windows hit a margin call<extra></extra>"),
    ]
    z3 = zd[Ld.index(3.0)][2]
    z2l = zl[Ll.index(2.0)][2]
    head = f"At 3× leverage a fall of half or more happens in {z3:.0f}% of 10-year windows"
    sub = ("Share of historical US-market windows 1926–2026 (daily rebalancing, no financing spread), "
           "and of windows with a margin call for a fixed loan (25% maintenance)")
    sub_n = "Share of US-market windows 1926–2026,<br>by leverage and horizon"

    def lay(narrow):
        if not narrow:
            xd, yd = [0, 0.47], [0, 0.88]
            xd2, yd2 = [0.55, 1], [0.0 + 0.88 * (1 - len(Ll) / len(Ld)), 0.88]
        else:
            xd, yd = [0, 1], [0.36, 0.89]
            xd2, yd2 = [0, 1], [0.0, 0.26]
        ann = [
            note("<b>Fall of 50% or more</b> at some close<br>daily rebalanced" if not narrow else "<b>Fall of 50% or more</b> (daily rebalanced)",
                 xd[0], yd[1], xref="paper", yref="paper", anchor="left", yanchor="bottom", yshift=8, size=12),
            note("<b>Margin call</b>, fixed loan<br>25% maintenance, forced closure" if not narrow else "<b>Margin call</b> (fixed loan, 25% maintenance)",
                 xd2[0], yd2[1], xref="paper", yref="paper", anchor="left", yanchor="bottom", yshift=8, size=12),
        ]
        return base_layout(head if not narrow else "A 50% fall is common<br>at 3× or more", sub if not narrow else sub_n,
                           500 if not narrow else 800,
                           {"l": 60, "r": 20, "t": 84, "b": 50} if not narrow else {"l": 46, "r": 12, "t": 118, "b": 40},
                           xaxis=ax(domain=xd, anchor="y", type="category", side="bottom", showline=False, showgrid=False, tickfont={"color": MUTE, "size": 12}),
                           xaxis2=ax(domain=xd2, anchor="y2", type="category", side="bottom", showline=False, showgrid=False, tickfont={"color": MUTE, "size": 12}),
                           yaxis=ax(domain=yd, anchor="x", type="category", showline=False, showgrid=False, tickfont={"color": MUTE, "size": 12}, title=ax_title("Leverage" if not narrow else "")),
                           yaxis2=ax(domain=yd2, anchor="x2", type="category", showline=False, showgrid=False, tickfont={"color": MUTE, "size": 12}, title=ax_title("Leverage" if not narrow else "")),
                           annotations=ann)

    wide, nar = lay(False), lay(True)
    finish_wrap("05-heatmap-fall", data, wide, nar, 1000,
                f"How often a leveraged position falls by half, and how often a loan is called, by leverage and horizon. At 3× daily-rebalanced, {z3:.0f}% of 10-year windows contain a fall of 50% or more; a fixed 2× loan with 25% maintenance is called in {z2l:.0f}% of 10-year windows.",
                f"Two heatmaps of percentages. Left: probability of a fall of 50% or more at some close, for daily-rebalanced leverage 0.5× to 6× (rows) and horizons of 1, 5, 10 and 20 years (columns); it grows with both, from {zd[0][0]:.0f}% at 0.5× over 1 year to {zd[-1][-1]:.0f}% at 6× over 20 years, and is {z3:.0f}% at 3× over 10 years. Right: probability of a margin call for a fixed loan with 25% maintenance, leverage 1.25× to 3×; at 2× it is {zl[Ll.index(2.0)][0]:.0f}% over 1 year and {z2l:.0f}% over 10 years, and at 3× {zl[-1][2]:.0f}% over 10 years.",
                SRC_US + " Windows start every 21st trading day and overlap; daily closes only, so intraday gaps are not modelled.")


# =================================================================================================================
# Figure 6: crash paths
# =================================================================================================================
EPISODES = [
    ("1929–1932", "1929-09-24", "1932-07-30", "Oct 1929 crash"),
    ("1987", "1987-09-18", "1988-01-29", "19 Oct 1987"),
    ("2007–2009", "2007-09-10", "2009-04-30", "financial crisis"),
    ("2020", "2020-01-20", "2020-06-30", "Covid crash"),
]


def episode_paths(a, b):
    d = US.loc[a:b]
    r, rf = d.MKT.values[1:], d.RF.values[1:]
    dates = list(d.index)
    paths = {l: 100 * daily_rebalanced(r, rf, l) for l in (1, 2, 3)}
    m = margin_loan(r, rf, 2.0, 0.25, 0.0, "close")
    loan = 100 * m.equity
    call = m.calls[0] if m.calls else None
    return dates, paths, loan, call


def fig6():
    eps = [(lab, *episode_paths(a, b), tag) for lab, a, b, tag in EPISODES]
    names = [("1× (no leverage)", BLUE, "solid", 1), ("2× daily rebalanced", GOLD, "solid", 2), ("3× daily rebalanced", ORANGE, "solid", 3)]
    data = []
    for i, (lab, dates, paths, loan, call, tag) in enumerate(eps):
        xa, ya = ("x" if i == 0 else f"x{i + 1}"), ("y" if i == 0 else f"y{i + 1}")
        ds = [dstr(d) for d in dates]
        for nm, col, dash, l in names:
            x, y = downsample(ds, paths[l], 2000)
            data.append({"type": "scatter", "mode": "lines", "name": nm, "legendgroup": nm, "showlegend": i == 0,
                         "xaxis": xa, "yaxis": ya, "x": x, "y": sigs(y),
                         "line": {"color": col, "width": 1.8},
                         "hovertemplate": f"%{{x|%d %b %Y}}: %{{y:.1f}}<extra>{nm}, {lab}</extra>"})
        x, y = downsample(ds, loan, 2000)
        nm = "2× with a loan (25% maintenance)"
        data.append({"type": "scatter", "mode": "lines", "name": nm, "legendgroup": nm, "showlegend": i == 0,
                     "xaxis": xa, "yaxis": ya, "x": x, "y": sigs(y), "line": {"color": PURPLE, "width": 1.8, "dash": "dash"},
                     "hovertemplate": f"%{{x|%d %b %Y}}: %{{y:.1f}}<extra>{nm}, {lab}</extra>"})
        if call is not None:
            data.append({"type": "scatter", "mode": "markers", "name": "Margin call", "legendgroup": "call", "showlegend": i == 0 or False,
                         "xaxis": xa, "yaxis": ya, "x": [ds[call]], "y": [sig(loan[call])],
                         "marker": {"symbol": "x", "size": 12, "color": PURPLE, "line": {"width": 3, "color": PURPLE}},
                         "hovertemplate": f"Margin call {fdate(dates[call])}: position closed, equity %{{y:.1f}}<extra></extra>"})
    # legend entry for the call marker: ensure one exists
    if not any(t.get("name") == "Margin call" and t.get("showlegend") for t in data):
        for t in data:
            if t.get("name") == "Margin call":
                t["showlegend"] = True
                break
    head = "Real crashes at 1×, 2× and 3×: leverage multiplies the fall, and a margin call locks it in"
    sub = "US market paths from about a month before each crash, 100 at the start, logarithmic vertical axis (each panel has its own scale)"
    sub_n = "US market paths from about a month before each crash,<br>100 = start, log scale (each panel its own scale)"

    def lay(narrow):
        if not narrow:
            cols = [[0, 0.46], [0.54, 1]]
            rows = [[0.55, 0.9], [0.0, 0.35]]
            doms = [(cols[0], rows[0]), (cols[1], rows[0]), (cols[0], rows[1]), (cols[1], rows[1])]
            h = 700
        else:
            doms = [([0, 1], [0.78 - 0.265 * 0 + 0.0, 0.93]), ([0, 1], [0.54, 0.69]), ([0, 1], [0.30, 0.45]), ([0, 1], [0.06, 0.21])]
            h = 1100
        axes, ann = {}, []
        for i, (lab, dates, paths, loan, call, tag) in enumerate(eps):
            sx, sy = ("" if i == 0 else str(i + 1)), ("" if i == 0 else str(i + 1))
            allv = np.concatenate([paths[1], paths[2], paths[3], loan])
            allv = allv[allv > 0]
            lo, hi = math.log10(allv.min() * 0.8), math.log10(allv.max() * 1.15)
            cand = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100] if hi - lo > 1.3 else [10, 15, 20, 30, 40, 50, 70, 100]
            ticks = [v for v in cand if 10 ** lo <= v <= 10 ** hi]
            span = (dates[-1] - dates[0]).days
            axes[f"xaxis{sx}"] = ax(domain=doms[i][0], anchor=f"y{sy}", type="date", tickformat="%b %Y", nticks=4 if not narrow else 5,
                                    range=[dstr(dates[0]), dstr(dates[-1])])
            axes[f"yaxis{sy}"] = ax(domain=doms[i][1], anchor=f"x{sx}", type="log", range=[lo, hi], tickvals=ticks,
                                    ticktext=[f"{t:g}" for t in ticks])
            ann.append(note(f"<b>{lab}</b>", doms[i][0][0], doms[i][1][1], xref="paper", yref="paper", anchor="left", yanchor="bottom",
                            yshift=6, size=13))
            xr, yr = f"x{sx}", f"y{sy}"
            msg = (f"Margin call {fdate(dates[call], True)}: forced out with {loan[call]:.0f}" if call is not None
                   else f"No margin call: the 2× loan ends at {loan[-1]:.0f}")
            ann.pop()
            ann.append(note(f"<b>{lab}</b>" + ("<br>" if narrow else "   ") + f"<span style='color:{PURPLE};font-size:12px'>{msg}</span>",
                            doms[i][0][0], doms[i][1][1], xref="paper", yref="paper", anchor="left", yanchor="bottom", yshift=6, size=13))
            lows = {l: float(paths[l].min()) for l in (1, 2, 3)}
            ann.append(note(f"Lowest: 1× {lows[1]:.0f}, 2× {lows[2]:.3g}, 3× {lows[3]:.2g}" if not narrow else f"Lows 1× {lows[1]:.0f} · 2× {lows[2]:.3g} · 3× {lows[3]:.2g}",
                            dstr(dates[-1]), 10 ** (hi - 0.02 * (hi - lo)), xref=xr, yref=yr, anchor="right", yanchor="top", color=MUTE, size=11, ylog=True))
        return base_layout(head if not narrow else "Real crashes at 1×, 2×, 3×:<br>leverage multiplies the fall",
                           sub if not narrow else sub_n, h,
                           {"l": 52, "r": 20, "t": 120, "b": 40} if not narrow else {"l": 46, "r": 12, "t": 216, "b": 36},
                           showlegend=True, narrow=narrow, annotations=ann, **axes,
                           legend={"font": {"size": 12 if not narrow else 11, "color": INK}, "orientation": "h", "x": 0, "xanchor": "left",
                                   "y": 1.0, "yanchor": "bottom", "bgcolor": "rgba(255,255,255,0)", "itemwidth": 30})

    wide, nar = lay(False), lay(True)
    calls = [(lab, fdate(dates[c]), float(loan[c])) for lab, dates, p, loan, c, t in eps if c is not None]
    nocall = [lab for lab, dates, p, loan, c, t in eps if c is None]
    mins = {lab: {l: float(p[l].min()) for l in (1, 2, 3)} for lab, d, p, lo_, c, t in eps}
    finish_wrap("06-crash-paths", data, wide, nar, 1400,
                f"Four real episodes, each starting about a month before the crash. At 3× the 1929–1932 episode shrinks 100 to {mins['1929–1932'][3]:.1f} and the 2007–2009 episode to {mins['2007–2009'][3]:.0f}; a fixed 2× loan was called in {' and '.join(c[0] for c in calls)} and survived {' and '.join(nocall)}.",
                "Four small charts on a logarithmic vertical axis, each starting at 100 about a month before a crash: 1929–1932, 1987, 2007–2009 and 2020. Each shows the market at 1×, and at 2× and 3× with daily rebalancing, and a 2× fixed loan with 25% maintenance margin. Lowest values at 1× / 2× / 3×: "
                + "; ".join(f"{lab} {m[1]:.0f} / {m[2]:.0f} / {m[3]:.1f}" for lab, m in mins.items())
                + ". The loan was called, and the position closed, on "
                + "; ".join(f"{d} ({lab}, equity {e:.0f})" for lab, d, e in calls) + f"; it was never called in {' or '.join(nocall)}.",
                SRC_US + " Daily closes; the loan position closes at the first close below 25% equity, and equity then earns the T-bill rate.")


# =================================================================================================================
# Figure 7: fat tails
# =================================================================================================================
def fig7():
    e = pd.read_csv(RES / "extreme_day_counts.csv")
    e = e[e["sample"] == "1926-2026"].sort_values("k")
    wd = pd.read_csv(RES / "worst_days.csv")
    wd = wd[(wd["sample"] == "1926-2026") & (wd["rank"] == 1)].iloc[0]
    n = int(pd.read_csv(RES / "down_streaks.csv").n_days.iloc[0])
    ks = [int(v) for v in e.k]
    exp = e.expected_each_side.values
    wait = n / exp / 252 * (n / n)                       # years: one such day every (n days / expected count) / 252
    # expected count over the whole sample = n * p, so wait (years) = 1 / (p * 252) = n / (exp * 252)
    def lab(w):
        if w < 1000:
            return f"1 in {w:,.0f} years" if w >= 10 else f"1 in {w:.1f} years"
        return f"1 in 10<sup>{round(math.log10(w))}</sup> years"
    zk = abs(float(wd.z_score))
    data = [
        {"type": "scatter", "mode": "lines+markers+text", "name": "Normal model (expected)", "x": ks, "y": [float(f"{v:.4g}") for v in exp],
         "line": {"color": INK, "width": 1.2, "dash": "dot"}, "marker": {"size": 8, "color": "white", "line": {"color": INK, "width": 1.5}},
         "text": [lab(w) for w in wait], "textposition": "middle left", "textfont": {"size": 11, "color": MUTE},
         "hovertemplate": "Beyond %{x} standard deviations: %{y:.2g} expected down days<extra>Normal model</extra>"},
        {"type": "scatter", "mode": "lines+markers", "name": "Observed down days", "x": ks, "y": [int(v) for v in e.observed_down],
         "line": {"color": BLUE, "width": 2}, "marker": {"size": 8, "color": BLUE, "line": {"color": "white", "width": 2}},
         "hovertemplate": "Beyond %{x} standard deviations: %{y:,} down days<extra>Observed</extra>"},
        {"type": "scatter", "mode": "markers", "name": "19 Oct 1987", "x": [sig(zk, 3)], "y": [1],
         "marker": {"size": 11, "color": BLUE, "line": {"color": "white", "width": 2}},
         "hovertemplate": f"{fdate(wd.date)}: {str(round(float(wd.MKT) * 100, 1)).replace('-', '−')}% in one day, {float(wd.z_score):.1f} standard deviations<extra></extra>"},
    ]
    o5 = int(e[e.k == 5].observed_down.iloc[0])
    e5 = float(e[e.k == 5].expected_each_side.iloc[0])
    head = f"The normal model says these days almost never happen; the market had {o5} beyond 5 standard deviations"
    sub = (f"US market down days beyond k standard deviations, {n:,} trading days 1926–2026, "
           "against a normal model with the same mean and spread")
    sub_n = f"US down days beyond k standard deviations, {n:,}<br>trading days 1926–2026, against a normal model"

    def lay(narrow):
        o10 = int(e[e.k == 10].observed_down.iloc[0])
        ann = [
            arrow(f"{fdate(wd.date)}: {('%.1f' % (float(wd.MKT) * 100)).replace('-', '−')}% in one day, {abs(zk):.0f} standard deviations;<br>the normal model: 1 in 10<sup>{round(float(wd.log10_wait_years_normal))}</sup> years" if not narrow else f"19 Oct 1987:<br>−{abs(float(wd.z_score)):.0f} SD",
                  zk, 1, -26 if not narrow else -8, -38 if not narrow else -44, anchor="right" if not narrow else "right", ylog=True),
            note("Observed", 3.7, 105, anchor="center", color=BLUE, yshift=14, xshift=-8, size=13) if False else
            note(f"Observed: {int(e.observed_down.iloc[1]):,} beyond 4, {o5} beyond 5, {o10} beyond 10", 6.2, 40, anchor="left", color=BLUE, size=12, yshift=26)
            if not narrow else {},
            note(f"Normal model: {e5:.3f} expected beyond 5" if not narrow else "", 5, e5, anchor="left", color=INK, size=11, xshift=0, yshift=-30) if False else {},
        ]
        ann = [a for a in ann if a]
        return base_layout(head if not narrow else f"Normal model: these days never<br>happen; the market had {o5} beyond 5 SD",
                           sub if not narrow else sub_n, 500,
                           {"l": 70, "r": 40, "t": 104, "b": 60} if not narrow else {"l": 56, "r": 12, "t": 200, "b": 54},
                           showlegend=True, narrow=narrow,
                           xaxis=ax(range=[1.4 if not narrow else -2.5, 17.4], tickvals=[3, 4, 5, 6, 8, 10, round(zk)],
                                    ticktext=["3", "4", "5", "6", "8", "10", f"{zk:.0f}"],
                                    title=ax_title("Standard deviations below the mean (k)")),
                           yaxis=ax(type="log", range=[-21.5, 3.3], tickvals=[1e-20, 1e-15, 1e-10, 1e-5, 1, 1e3],
                                    ticktext=["10<sup>−20</sup>", "10<sup>−15</sup>", "10<sup>−10</sup>", "10<sup>−5</sup>", "1", "1,000"],
                                    title=ax_title("Down days beyond k (log scale)" if not narrow else "")),
                           annotations=ann,
                           legend={"font": {"size": 12, "color": INK}, "orientation": "h", "x": 0, "xanchor": "left", "y": 1.0,
                                   "yanchor": "bottom", "bgcolor": "rgba(255,255,255,0)", "itemwidth": 30})

    wide, nar = lay(False), lay(True)
    finish_wrap("07-fat-tails", data, wide, nar, 1000,
                f"Days on which the US market fell by more than k standard deviations, against the count expected under a normal model with the same mean and spread. Beyond 5 standard deviations there were {o5} such days against {e5:.3f} expected; the 19 October 1987 fall ({float(wd.MKT) * 100:.1f}%) was {abs(zk):.0f} standard deviations.",
                f"Chart on a logarithmic vertical axis of the number of down days beyond k standard deviations, k = 3, 4, 5, 6, 8, 10. Observed: {', '.join(str(int(v)) for v in e.observed_down)}. Normal-model expectation: {', '.join(f'{v:.2g}' for v in exp)}, labelled as one such day in {', '.join(lab(w).replace('1 in ', '').replace('<sup>', '^').replace('</sup>', '') for w in wait)}. A separate point marks 19 October 1987, {abs(zk):.1f} standard deviations below the mean, which the normal model would expect once in about 10^{wd.log10_wait_years_normal:.0f} years.",
                SRC_US + " Standard deviation and mean are those of the full sample; down days only.")


# =================================================================================================================
# Figure 8: the early short
# =================================================================================================================
def fig8():
    ctx = pd.read_csv(RES / "short_1996_context.csv").iloc[0]
    mg = pd.read_csv(RES / "short_1996_margin.csv")
    mg = mg[(mg.action == "close") & (mg.horizon_days == 7500)].iloc[0]
    entry = pd.Timestamp(ctx.entry_close_date)
    idx = (1 + US.MKT).cumprod()
    x = (idx / idx.loc[entry]).loc[entry:"2004-12-31"]
    peak, low = pd.Timestamp(ctx.dotcom_peak_date_by_end_2003), pd.Timestamp(ctx.lowest_after_that_peak_date)
    call = pd.Timestamp(mg.first_call_date)
    assert abs(x.loc[peak] - ctx.dotcom_peak_vs_entry) < 1e-6 and abs(x.loc[low] - ctx.lowest_after_that_peak_vs_entry) < 1e-6
    assert abs(x.loc[call] - 1 - mg.index_rise_at_first_call) < 1e-6
    last_below = x[x <= 1.0].index[-1]
    xs, ys = downsample([dstr(d) for d in x.index], x.values, 2000)
    ys = np.asarray(ys)
    data = [
        {"type": "scatter", "mode": "lines", "x": [xs[0], xs[-1]], "y": [1, 1], "line": {"color": INK, "width": 1.2, "dash": "dash"},
         "hoverinfo": "skip", "showlegend": False},
        {"type": "scatter", "mode": "lines", "name": "Short under water", "x": xs, "y": sigs(np.maximum(ys, 1.0)), "fill": "tonexty",
         "fillcolor": "rgba(201,111,51,0.22)", "line": {"width": 0}, "hoverinfo": "skip"},
        {"type": "scatter", "mode": "lines", "name": "US market, total-return index", "x": xs, "y": sigs(ys),
         "line": {"color": BLUE, "width": 2}, "hovertemplate": "%{x|%d %b %Y}: %{y:.2f} × entry<extra></extra>"},
        {"type": "scatter", "mode": "markers", "name": "Marked days", "x": [dstr(call), dstr(peak), dstr(low)],
         "y": sigs([x.loc[call], x.loc[peak], x.loc[low]]),
         "marker": {"size": 10, "color": [PURPLE, ORANGE, GOLD], "line": {"color": "white", "width": 2}}, "hoverinfo": "skip",
         "showlegend": False},
    ]
    head = "The bubble burst, but the market never fell back to where the December 1996 short was sold"
    sub = (f"US total-return index relative to the close of {fdate(entry)} (= 1); shaded: the short would be under water; "
           f"it last stood at or below 1 on {fdate(last_below)}")
    sub_n = f"US total-return index, 5 Dec 1996 = 1;<br>shaded: a 1× short would be under water"

    def lay(narrow):
        ann = [
            arrow(f"{fdate(call)}: margin call<br>(30% maintenance; index up {pct(mg.index_rise_at_first_call)},<br>equity down to {pct(mg.equity_at_first_call)})" if not narrow else f"{fdate(call, True)}:<br>margin call",
                  dstr(call), float(x.loc[call]), -30 if not narrow else -6, -40 if not narrow else -30, anchor="right", color=PURPLE),
            arrow(f"{fdate(peak)}: peak, +{100 * (x.loc[peak] - 1):.0f}%" if not narrow else f"{fdate(peak, True)}:<br>+{100 * (x.loc[peak] - 1):.0f}%",
                  dstr(peak), float(x.loc[peak]), 10, -22, anchor="left" if not narrow else "center", color=ORANGE),
            arrow(f"{fdate(low)}: low,<br>still +{100 * (x.loc[low] - 1):.0f}% above entry" if not narrow else f"{fdate(low, True)}:<br>still +{100 * (x.loc[low] - 1):.0f}%",
                  dstr(low), float(x.loc[low]), 24 if not narrow else 0, 40 if not narrow else 40, anchor="left" if not narrow else "center", color=GOLD),
            note("Entry: 1", dstr(x.index[0]), 1, anchor="left", yanchor="top", yshift=-6, xshift=4, color=INK, size=11),
        ]
        return base_layout(head if not narrow else "The bubble burst, but the market never<br>fell back to the short's entry",
                           sub if not narrow else sub_n, 500,
                           {"l": 56, "r": 36, "t": 84, "b": 56} if not narrow else {"l": 44, "r": 12, "t": 180, "b": 50},
                           showlegend=narrow, narrow=narrow,
                           xaxis=ax(type="date", range=[dstr(x.index[0]), "2004-12-31"], tickformat="%Y", dtick="M12" if not narrow else "M24"),
                           yaxis=ax(range=[0.8, 2.5], tickvals=[1, 1.5, 2, 2.5], ticktext=["1", "1.5", "2", "2.5"],
                                    title=ax_title("Index relative to entry" if not narrow else "")),
                           annotations=ann)

    wide, nar = lay(False), lay(True)
    finish_wrap("08-early-short", data, wide, nar, 1000,
                f"The US stock market relative to 5 December 1996, the day of Greenspan's “irrational exuberance” speech. A short position opened that day was called on {fdate(call)} (index up {pct(mg.index_rise_at_first_call)}), the market peaked {pct(x.loc[peak] - 1)} above entry on {fdate(peak)}, and even the October 2002 low was {pct(x.loc[low] - 1)} above it.",
                f"Line chart of the US total-return index from 5 December 1996 (= 1) to the end of 2004. It rises above 1 from early 1997, reaches {x.loc[peak]:.2f} on {fdate(peak)} and falls to {x.loc[low]:.2f} on {fdate(low)}, never back to 1; the area between the line and 1 is shaded. Marked: the margin call of a 1× short on {fdate(call)} when the index stood at {x.loc[call]:.2f}, the peak, and the 2002 low.",
                SRC_US + " Index built from daily returns including dividends (a short pays them); margin call at 30% maintenance, checked at closes.")


# =================================================================================================================
def write_captions():
    lines = ["# Captions, alt text and sources: risk-ruin figures", "",
             "Generated by `scripts/make_figures.py`; every number is read from `results/` or recomputed from the raw data.", ""]
    import re
    fix = lambda t: re.sub(r"(?<![\w.])-(?=\d)", "−", t)
    for name, cap, alt, src, ratio_h in CAPTIONS:
        cap, alt = fix(cap), fix(alt)
        lines += [f"## {name}", "", f"- **Caption:** {cap}", f"- **Alt text:** {alt}", f"- **Source line:** {src}",
                  f"- **Image size:** 2200 x {ratio_h} (suggested `--pf-ratio:2200 / {ratio_h}`)", ""]
    (FIGS / "CAPTIONS.md").write_text("\n".join(lines))


if __name__ == "__main__":
    only = [a for a in sys.argv[1:] if not a.startswith("--")]
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8):
        if not only or any(f.__name__ == "fig" + o.lstrip("0") for o in only):
            f()
    write_captions()
