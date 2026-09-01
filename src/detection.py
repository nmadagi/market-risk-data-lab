"""Statistical anomaly detection over risk factor series.

Three detectors, one principle each:
- stale: real markets move; a run of zero returns is a frozen feed
- spike: size every move in the series' own recent volatility (EWMA)
- gap: completeness against the expected business calendar

Two tiebreakers decide what a big move actually is, because "big" alone
is not evidence of an error:
- peers: did correlated series move the same day? Then it is a market
  event, not a feed problem.
- reversal: a corrupt print is followed by an equal and opposite move as
  the feed returns to reality. A genuine regime move or a vendor level
  shift is not. So spike plus reversal is a data error; spike without
  reversal and without peers is a level break that a human should look at.
"""
import numpy as np
import pandas as pd

PEERS = {
    "usd2y": ["usd5y", "usd10y"],
    "usd5y": ["usd2y", "usd10y"],
    "usd10y": ["usd2y", "usd5y"],
    "swaption_vol": ["usd5y", "usd10y"],
    "eurusd": [],
    "credit_spread": [],
}

SPIKE_WARMUP = 60          # days of history before the vol estimate is trusted
REVERSAL_MIN = 0.5         # next-day move must undo at least half the spike

VERDICT_ERROR = "likely data error"
VERDICT_REAL = "likely real move"
VERDICT_BREAK = "level break, review"


def detect_stale(s: pd.Series, min_run: int = 5):
    """Runs of >= min_run identical consecutive values."""
    same = s.diff() == 0
    findings = []
    run_start, run_len = None, 0
    for date, flag in same.items():
        if flag:
            run_len += 1
            run_start = run_start or date
        else:
            if run_len >= min_run:
                findings.append(_f("stale", s.name, run_start, run_len,
                                   f"{run_len} identical prints"))
            run_start, run_len = None, 0
    if run_len >= min_run:
        findings.append(_f("stale", s.name, run_start, run_len,
                           f"{run_len} identical prints"))
    return findings


def detect_spikes(s: pd.Series, z_thresh: float = 6.0, lam: float = 0.94):
    """Daily move vs EWMA volatility of the series' own history.

    The first SPIKE_WARMUP days are skipped: with almost no history the
    volatility estimate is noise and everything looks like a spike.
    """
    d = s.diff()
    ewma_var = d.pow(2).ewm(alpha=1 - lam).mean().shift(1)
    z = d / np.sqrt(ewma_var)
    z.iloc[:SPIKE_WARMUP] = np.nan
    hits = z.abs() > z_thresh
    return [_f("spike", s.name, date, 1, f"move of {z[date]:.1f} sigma")
            for date in s.index[hits.fillna(False)]]


def detect_gaps(s: pd.Series):
    """Missing values inside the series' own live range."""
    live = s.loc[s.first_valid_index():s.last_valid_index()]
    missing = live[live.isna()]
    if missing.empty:
        return []
    return [_f("gap", s.name, missing.index[0], len(missing),
               f"{len(missing)} missing days")]


def peer_confirms(df: pd.DataFrame, col: str, date, z_thresh: float = 3.0) -> bool:
    """True if correlated peers also moved big that day (real event, not error)."""
    for p in PEERS.get(col, []):
        d = df[p].diff()
        sigma = d.rolling(60).std().shift(1)
        z = d / sigma
        if date in z.index and abs(z.get(date, 0)) > z_thresh:
            return True
    return False


def reverses_next_day(s: pd.Series, date) -> bool:
    """True if the following day's move undoes most of this day's move."""
    d = s.diff()
    i = s.index.get_loc(date)
    if i + 1 >= len(s) or pd.isna(d.iloc[i]) or pd.isna(d.iloc[i + 1]):
        return False
    return (-d.iloc[i + 1] / d.iloc[i]) >= REVERSAL_MIN


def run_all(df: pd.DataFrame) -> pd.DataFrame:
    """All detectors on all series; spikes get both tiebreakers."""
    findings = []
    for col in df.columns:
        findings += detect_stale(df[col])
        findings += detect_gaps(df[col])
        for f in detect_spikes(df[col]):
            f["peer_confirmed"] = peer_confirms(df, col, f["start"])
            f["reverses"] = reverses_next_day(df[col], f["start"])
            if f["peer_confirmed"]:
                f["verdict"] = VERDICT_REAL
            elif f["reverses"]:
                f["verdict"] = VERDICT_ERROR
            else:
                f["verdict"] = VERDICT_BREAK
            findings.append(f)
    out = pd.DataFrame(findings)
    return out.sort_values("start").reset_index(drop=True) if not out.empty else out


def _f(kind, series, start, length, detail):
    return {"type": kind, "series": series, "start": start,
            "length": length, "detail": detail}
