"""Statistical anomaly detection over risk factor series.

Three detectors, one principle each:
- stale: real markets move; a run of zero returns is a frozen feed
- spike: size every move in the series' own recent volatility (EWMA)
- gap: completeness against the expected business calendar
Plus the tiebreaker for spike vs real move: did correlated peers move too?
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
    """Daily move vs EWMA volatility of the series' own history."""
    d = s.diff()
    ewma_var = d.pow(2).ewm(alpha=1 - lam).mean().shift(1)
    z = d / np.sqrt(ewma_var)
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
    peers = PEERS.get(col, [])
    for p in peers:
        d = df[p].diff()
        sigma = d.rolling(60).std().shift(1)
        z = d / sigma
        if date in z.index and abs(z.get(date, 0)) > z_thresh:
            return True
    return False


def run_all(df: pd.DataFrame) -> pd.DataFrame:
    """All detectors on all series; spikes get the cross sectional check."""
    findings = []
    for col in df.columns:
        findings += detect_stale(df[col])
        findings += detect_gaps(df[col])
        for f in detect_spikes(df[col]):
            f["peer_confirmed"] = peer_confirms(df, col, f["start"])
            f["verdict"] = "likely real move" if f["peer_confirmed"] else "likely data error"
            findings.append(f)
    out = pd.DataFrame(findings)
    return out.sort_values("start").reset_index(drop=True) if not out.empty else out


def _f(kind, series, start, length, detail):
    return {"type": kind, "series": series, "start": start,
            "length": length, "detail": detail}
