"""Remediation: propose fixes, check them, never touch the golden copy blindly.

The design principle: the proposal engine proposes, deterministic guardrails
dispose. Fixes land in a staging copy, every repaired point is flagged, and
promotion requires the guardrail checks to pass. Material VaR impact routes
to human review instead of auto-accept.

The fix ladder, ordered by trust:
  1 day        -> drop and interpolate
  short gap    -> linear interpolation between good neighbors
  long stretch -> proxy regression on correlated peer series
"""
import numpy as np
import pandas as pd
from scipy import stats

from src.detection import PEERS
from src import risk

SHORT_GAP_MAX = 5
KS_PVALUE_MIN = 0.05
VAR_IMPACT_REVIEW_PCT = 5.0


def propose(corrupted: pd.DataFrame, findings: pd.DataFrame) -> list:
    """One proposal per finding that looks like a data error."""
    proposals = []
    for _, f in findings.iterrows():
        if f.get("verdict") == "likely real move":
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


def apply_to_staging(corrupted: pd.DataFrame, proposals: list):
    """Apply proposals to a staging copy; return (staged_df, flags_df)."""
    staged = corrupted.copy()
    flags = []
    for p in proposals:
        for d, v in zip(p["dates"], p["values"]):
            staged.loc[d, p["series"]] = v
            flags.append({"date": d, "series": p["series"],
                          "method": p["method"], "filled": True})
    return staged, pd.DataFrame(flags)


def guardrail_check(clean_ref: pd.DataFrame, staged: pd.DataFrame,
                    corrupted: pd.DataFrame, proposal: dict) -> dict:
    """Deterministic acceptance checks for one proposal.

    ks_pvalue     : filled-region returns vs the series' observed returns.
                    Low p means the repair changed the distribution shape.
    var_impact_pct: how much the repair moves 99 pct VaR vs the corrupted
                    input. Material impact is not a rejection, it is a
                    routing decision: a human signs off, the pipeline does not.
    """
    series = proposal["series"]
    dates = proposal["dates"]
    lo = max(0, staged.index.get_loc(dates[0]) - 30)
    hi = min(len(staged) - 1, staged.index.get_loc(dates[-1]) + 30)
    window = staged[series].iloc[lo:hi + 1]
    region_ret = window.diff().dropna()
    obs_ret = clean_ref[series].diff().dropna()
    ks_p = stats.ks_2samp(region_ret, obs_ret.sample(
        min(len(obs_ret), 250), random_state=0)).pvalue

    var_before = risk.var99(risk.pnl_vector(corrupted.ffill()))
    var_after = risk.var99(risk.pnl_vector(staged))
    impact = abs(var_after - var_before) / abs(var_before) * 100

    needs_review = impact > VAR_IMPACT_REVIEW_PCT
    accepted = ks_p >= KS_PVALUE_MIN and not needs_review
    return {"series": series, "method": proposal["method"],
            "ks_pvalue": round(float(ks_p), 4),
            "var_impact_pct": round(float(impact), 2),
            "needs_review": needs_review, "accepted": accepted}


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
    """Regress the target on correlated peers over clean history, predict the hole."""
    peers = [p for p in PEERS.get(series, []) if p in df.columns]
    if not peers:
        return None
    good = df.drop(index=dates).dropna(subset=[series] + peers)
    X = np.column_stack([good[p] for p in peers] + [np.ones(len(good))])
    beta, *_ = np.linalg.lstsq(X, good[series].to_numpy(), rcond=None)
    Xh = np.column_stack([df.loc[dates, p] for p in peers] + [np.ones(len(dates))])
    return (Xh @ beta).tolist()


def _rationale(kind, method, n):
    if method == "interpolate":
        return f"{kind}: {n} points dropped and linearly interpolated from good neighbors"
    return f"{kind}: {n} points rebuilt by regression on correlated peer series"
