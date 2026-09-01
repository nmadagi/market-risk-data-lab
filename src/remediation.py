"""Remediation: propose fixes, check each one, promote only what passes.

The design principle: the proposal engine proposes, deterministic guardrails
dispose. Every proposal is scored on its own, against only the data the
pipeline would really have (no peeking at a clean truth), and only the
accepted ones are applied to the staging copy. Rejected ones are kept for
the audit trail. Material VaR impact routes to human review instead of
auto-accept. Every applied point is flagged, and the golden copy is never
edited in place.

The fix ladder, ordered by trust:
  1 day        -> drop and interpolate
  short gap    -> linear interpolation between good neighbors
  long stretch -> regression on correlated peer series, in change space
"""
import numpy as np
import pandas as pd
from scipy import stats

from src.detection import PEERS, VERDICT_ERROR
from src import risk

SHORT_GAP_MAX = 5
KS_PVALUE_MIN = 0.05
VAR_IMPACT_REVIEW_PCT = 5.0
KS_WINDOW = 30


def propose(corrupted: pd.DataFrame, findings: pd.DataFrame) -> list:
    """One proposal per finding that looks like a data error.

    Spikes are proposed only when the verdict is a data error. Real moves
    and level breaks are left alone: a level break needs a human to decide
    whether it is a vendor splice or a genuine repricing, and interpolating
    it away would destroy real history.
    """
    proposals = []
    for _, f in findings.iterrows():
        if f["type"] == "spike" and f.get("verdict") != VERDICT_ERROR:
            continue
        series, kind = f["series"], f["type"]
        dates = _affected_dates(corrupted, f)
        if len(dates) == 0:
            continue
        if kind == "spike" or len(dates) <= SHORT_GAP_MAX:
            method = "interpolate"
            values = _interpolate(corrupted[series], dates)
        else:
            method = "proxy_regression"
            values = _proxy_fill(corrupted, series, dates)
            if values is None:
                method = "interpolate"
                values = _interpolate(corrupted[series], dates)
        proposals.append({
            "series": series, "type": kind, "method": method,
            "dates": list(dates), "values": values,
            "rationale": _rationale(kind, method, len(dates)),
        })
    return proposals


def guardrail_check(base: pd.DataFrame, proposal: dict) -> dict:
    """Deterministic acceptance checks for ONE proposal, applied alone.

    base is the working frame the pipeline actually has (corrupted, with
    gaps carried forward so the risk engine can run). No clean reference
    is used anywhere: production never has one.

    ks_pvalue     : returns in a window around the repair vs this series'
                    own returns everywhere else. Low p means the repair
                    changed the local distribution shape.
    var_impact_pct: how much this repair alone moves 99 pct VaR. Material
                    impact is not a rejection, it is a routing decision:
                    a human signs off, the pipeline does not.
    """
    series, dates = proposal["series"], proposal["dates"]
    trial = _apply_one(base, proposal)

    lo = max(0, trial.index.get_loc(dates[0]) - KS_WINDOW)
    hi = min(len(trial) - 1, trial.index.get_loc(dates[-1]) + KS_WINDOW)
    region_ret = trial[series].iloc[lo:hi + 1].diff().dropna()
    outside = trial[series].drop(trial.index[lo:hi + 1]).diff().dropna()
    ks_p = stats.ks_2samp(region_ret, outside).pvalue

    var_before = risk.var99(risk.pnl_vector(base))
    var_after = risk.var99(risk.pnl_vector(trial))
    impact = abs(var_after - var_before) / abs(var_before) * 100

    needs_review = impact > VAR_IMPACT_REVIEW_PCT
    accepted = ks_p >= KS_PVALUE_MIN and not needs_review
    return {"series": series, "method": proposal["method"],
            "type": proposal["type"], "points": len(dates),
            "ks_pvalue": round(float(ks_p), 4),
            "var_impact_pct": round(float(impact), 2),
            "needs_review": bool(needs_review), "accepted": bool(accepted)}


def run(corrupted: pd.DataFrame, findings: pd.DataFrame):
    """Full loop: propose, check each alone, apply only the accepted.

    Returns (staged, flags, proposals, checks). staged starts from the
    working frame (gaps carried forward) and contains only accepted
    repairs; flags lists every applied point with its method.
    """
    base = corrupted.ffill()
    proposals = propose(corrupted, findings)
    checks = [guardrail_check(base, p) for p in proposals]
    accepted = [p for p, c in zip(proposals, checks) if c["accepted"]]
    staged, flags = apply_to_staging(base, accepted)
    return staged, flags, proposals, checks


def apply_to_staging(base: pd.DataFrame, proposals: list):
    """Apply proposals to a copy; return (staged_df, flags_df)."""
    staged = base.copy()
    flags = []
    for p in proposals:
        for d, v in zip(p["dates"], p["values"]):
            staged.loc[d, p["series"]] = v
            flags.append({"date": d, "series": p["series"],
                          "method": p["method"], "filled": True})
    cols = ["date", "series", "method", "filled"]
    return staged, pd.DataFrame(flags, columns=cols)


def _apply_one(base, proposal):
    out = base.copy()
    out.loc[proposal["dates"], proposal["series"]] = proposal["values"]
    return out


def _affected_dates(df, finding):
    kind = finding["type"]
    if kind == "gap":
        s = df[finding["series"]]
        return s[s.isna()].index
    start = finding["start"]
    length = int(finding["length"])
    if kind == "stale":
        # the frozen prints after the first (real) one
        return df.loc[start:].index[1:length + 1]
    return df.loc[start:].index[:length]


def _interpolate(series, dates):
    s = series.copy()
    s.loc[dates] = np.nan
    return s.interpolate(method="linear", limit_direction="both").loc[dates].tolist()


def _proxy_fill(df, series, dates):
    """Rebuild a stretch from correlated peers, in CHANGE space.

    Regressing levels looks fine on a chart and is wrong for risk: the
    fitted level does not meet the last real observation, so the repair
    injects an artificial jump on its first day, and a jump is exactly
    what a risk model reads as a loss. So: regress daily changes on peer
    daily changes, then walk forward from the last good value. A linear
    bridge spreads any residual mismatch across the gap so the far edge
    lands cleanly on the next real observation instead of jumping again.
    """
    peers = [p for p in PEERS.get(series, []) if p in df.columns]
    if not peers:
        return None
    dates = pd.DatetimeIndex(dates)
    chg = df[[series] + peers].diff()
    good = chg.drop(index=dates, errors="ignore").dropna()
    if len(good) < 30:
        return None
    X = np.column_stack([good[p] for p in peers] + [np.ones(len(good))])
    beta, *_ = np.linalg.lstsq(X, good[series].to_numpy(), rcond=None)

    if df.loc[dates, peers].isna().any().any():
        return None
    Xh = np.column_stack([chg.loc[dates, p].fillna(0.0) for p in peers]
                         + [np.ones(len(dates))])
    steps = Xh @ beta

    anchor = _last_good_before(df[series], dates[0])
    if anchor is None:
        return None
    path = anchor + np.cumsum(steps)

    nxt = _first_good_after(df[series], dates[-1])
    if nxt is not None:
        err = nxt - path[-1]
        n = len(path)
        path = path + err * (np.arange(1, n + 1) / (n + 1))
    return path.tolist()


def _last_good_before(s, date):
    prior = s.loc[:date].iloc[:-1].dropna()
    return float(prior.iloc[-1]) if len(prior) else None


def _first_good_after(s, date):
    later = s.loc[date:].iloc[1:].dropna()
    return float(later.iloc[0]) if len(later) else None


def _rationale(kind, method, n):
    if method == "interpolate":
        return f"{kind}: {n} point{'s' if n > 1 else ''} dropped and linearly interpolated from good neighbors"
    return f"{kind}: {n} points rebuilt by regression on correlated peer series"
