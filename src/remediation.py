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
MIN_AUTO_DAYS = 2          # single-day calls always need a person
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
                    changed the local distribution shape, and the repair is
                    rejected outright.
    needs_review  : the repair still happens, but a person signs it off
                    rather than the pipeline. Triggered by a material VaR
                    impact, or by the repair covering a single day. The
                    single-day rule comes straight from measurement: the
                    detector is essentially perfect on sustained faults and
                    around 60 pct on one-day events, and a one-day call is
                    where an automatic repair is most likely to smooth away
                    a real market move.
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

    accepted = ks_p >= KS_PVALUE_MIN
    needs_review = accepted and (impact > VAR_IMPACT_REVIEW_PCT
                                 or len(dates) < MIN_AUTO_DAYS)
    return {"series": series, "method": proposal["method"],
            "type": proposal["type"], "points": len(dates),
            "ks_pvalue": round(float(ks_p), 4),
            "var_impact_pct": round(float(impact), 2),
            "needs_review": bool(needs_review), "accepted": bool(accepted)}


def run(corrupted: pd.DataFrame, findings: pd.DataFrame, reviewer=None):
    """Full loop: propose, check each in turn, apply only the accepted.

    Returns (staged, flags, proposals, checks, unresolved).

    `reviewer` stands in for the person who signs off the repairs the
    guardrails route to review. It is given a proposal and returns True to
    approve. Without one every routed repair is treated as approved.

    Proposals are scored in date order against the data as it stands, so a
    repair is judged in the context of the repairs already accepted before
    it. Scoring every proposal against the original broken frame instead
    let one fault contaminate another's test window: a bad print 26 days
    after a frozen feed was rejected because the frozen days sat inside the
    window its distribution check looked at.

    staged starts from the working frame. Points with no accepted repair
    are carried forward so the risk engine can run at all, but that is a
    stopgap, not a fix: carry forward is the method this project's own
    evaluation shows flattens volatility. So every such point is listed in
    `unresolved` and stays on the exception report until a human deals
    with it. The pipeline never silently fills.
    """
    base = corrupted.ffill()
    proposals = sorted(propose(corrupted, findings), key=lambda p: p["dates"][0])
    staged = base.copy()
    checks, accepted = [], []
    for p in proposals:
        c = guardrail_check(staged, p)
        if c["needs_review"] and reviewer is not None:
            c["approved_at_review"] = bool(reviewer(p))
            c["accepted"] = c["accepted"] and c["approved_at_review"]
        checks.append(c)
        if c["accepted"]:
            staged = _apply_one(staged, p)
            accepted.append(p)
    _, flags = apply_to_staging(base, accepted)
    unresolved = _unresolved(corrupted, proposals, checks)
    return staged, flags, proposals, checks, unresolved


def _unresolved(corrupted, proposals, checks):
    """Every faulty point that did NOT get an accepted repair."""
    repaired = {(p["series"], d)
                for p, c in zip(proposals, checks) if c["accepted"]
                for d in p["dates"]}
    rejected_reason = {}
    for p, c in zip(proposals, checks):
        if c["accepted"]:
            continue
        why = ("repair routed to human review on VaR impact"
               if c["needs_review"] else
               f"repair rejected by distribution check (ks p={c['ks_pvalue']})")
        for d in p["dates"]:
            rejected_reason[(p["series"], d)] = f"{p['type']}: {why}"

    rows = []
    for col in corrupted.columns:
        for d in corrupted.index[corrupted[col].isna()]:
            if (col, d) in repaired:
                continue
            rows.append({"date": d, "series": col,
                         "reason": rejected_reason.get(
                             (col, d), "missing: no repair proposed"),
                         "value_in_use": "carried forward (stopgap)"})
    for (col, d), why in rejected_reason.items():
        if (col, d) in repaired or pd.isna(corrupted.loc[d, col]):
            continue
        rows.append({"date": d, "series": col, "reason": why,
                     "value_in_use": "original faulty value retained"})
    cols = ["date", "series", "reason", "value_in_use"]
    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values(["series", "date"]).reset_index(drop=True)


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
    # stale: `start` is the first frozen print and `length` counts them
    return df.loc[start:].index[:length]


def _interpolate(series, dates):
    s = series.copy()
    s.loc[dates] = np.nan
    return s.interpolate(method="linear", limit_direction="both").loc[dates].tolist()


def fit_proxy(df, series, exclude=None):
    """OLS of this series' daily changes on its peers' daily changes."""
    peers = _peers_of(df, series)
    if not peers:
        return None
    chg = df[[series] + peers].diff()
    good = chg.drop(index=exclude, errors="ignore").dropna() if exclude is not None \
        else chg.dropna()
    if len(good) < 30:
        return None
    X = np.column_stack([good[p] for p in peers] + [np.ones(len(good))])
    beta, *_ = np.linalg.lstsq(X, good[series].to_numpy(), rcond=None)
    return beta


def _proxy_fill(df, series, dates, model=None):
    """Rebuild a stretch from correlated peers, in CHANGE space.

    Regressing levels looks fine on a chart and is wrong for risk: the
    fitted level does not meet the last real observation, so the repair
    injects an artificial jump on its first day, and a jump is exactly
    what a risk model reads as a loss. So: regress daily changes on peer
    daily changes, then walk forward from the last good value. A linear
    bridge spreads any residual mismatch across the gap so the far edge
    lands cleanly on the next real observation instead of jumping again.
    """
    peers = _peers_of(df, series)
    if not peers:
        return None
    dates = pd.DatetimeIndex(dates)
    beta = model if model is not None else fit_proxy(df, series, exclude=dates)
    if beta is None:
        return None
    if df.loc[dates, peers].isna().any().any():
        return None
    chg = df[peers].diff().loc[dates].fillna(0.0)
    Xh = np.column_stack([chg[p] for p in peers] + [np.ones(len(dates))])
    return _anchor_and_bridge(df[series], dates, Xh @ beta)


def fit_ml(df, series, exclude=None):
    """Fit the forest once on everything outside the outages.

    Separated from the fill so a caller repairing several outages fits one
    model and applies it to each, which is both faster and the right
    methodology: one model, many outages, not a model per hole.
    """
    try:
        from sklearn.ensemble import RandomForestRegressor
    except ImportError:
        return None
    peers = _peers_of(df, series)
    if not peers:
        return None
    chg = df[[series] + peers].diff()
    good = chg.drop(index=exclude, errors="ignore").dropna() if exclude is not None \
        else chg.dropna()
    if len(good) < 100:
        return None
    model = RandomForestRegressor(n_estimators=100, min_samples_leaf=5,
                                  random_state=0, n_jobs=-1)
    model.fit(good[peers].to_numpy(), good[series].to_numpy())
    return model


def _ml_fill(df, series, dates, model=None):
    """Same job as _proxy_fill, with a random forest instead of a line.

    Identical inputs (peer daily changes) and identical anchoring, so the
    comparison in the evaluation harness is apples to apples and the only
    thing that varies is the functional form. A tree ensemble can capture
    a non-linear or regime-dependent relationship that OLS cannot. Whether
    it is worth the loss of interpretability is a question for the
    benchmark, not for taste: see evaluation.mask_and_recover.
    """
    peers = _peers_of(df, series)
    if not peers:
        return None
    dates = pd.DatetimeIndex(dates)
    if model is None:
        model = fit_ml(df, series, exclude=dates)
    if model is None:
        return None
    if df.loc[dates, peers].isna().any().any():
        return None
    steps = model.predict(df[peers].diff().loc[dates].fillna(0.0).to_numpy())
    return _anchor_and_bridge(df[series], dates, steps)


def _peers_of(df, series):
    return [p for p in PEERS.get(series, []) if p in df.columns]


def _anchor_and_bridge(s, dates, steps):
    """Walk predicted changes forward from the last real value, then
    spread any residual mismatch so the far edge lands on the next real
    observation instead of jumping."""
    anchor = _last_good_before(s, dates[0])
    if anchor is None:
        return None
    path = anchor + np.cumsum(steps)
    nxt = _first_good_after(s, dates[-1])
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
