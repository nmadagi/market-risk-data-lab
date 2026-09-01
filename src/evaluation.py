"""Mask and recover: the harness that proves a fill method deserves trust.

Hide points we actually observed, ask each method to reconstruct them,
score against the truth. Three scores, because average accuracy is not
enough for risk data:
  mae         : plain accuracy vs truth
  ks_pvalue   : does the repaired region keep the return distribution shape
  tail_ratio  : repaired vol of the region / true vol. Below 1 means the
                method smooths, and smoothed history understates VaR and
                weakens stress calibration. The most important number here.
Baseline method is interpolation: if the fancy method cannot beat it,
ship the simple one. That is why a random forest is in the lineup. It
gets exactly the same inputs as the linear proxy, so the benchmark
isolates the functional form, and it earns its place only if the numbers
say so. Measuring beats assuming in both directions.
"""
import numpy as np
import pandas as pd
from scipy import stats

from src.remediation import _ml_fill, _proxy_fill, fit_ml, fit_proxy


def _mask(series: pd.Series, frac: float, seed: int, block: int = 5):
    """Hide contiguous blocks (realistic: outages come in runs, not dots)."""
    rng = np.random.default_rng(seed)
    n_blocks = max(1, int(len(series) * frac / block))
    hidden = []
    for _ in range(n_blocks):
        i = rng.integers(50, len(series) - 50 - block)
        hidden.extend(series.index[i:i + block])
    return pd.DatetimeIndex(sorted(set(hidden)))


def _fill_interpolate(df, col, dates):
    s = df[col].copy()
    s.loc[dates] = np.nan
    return s.interpolate(method="linear").loc[dates]


def _fill_carry(df, col, dates):
    s = df[col].copy()
    s.loc[dates] = np.nan
    return s.ffill().loc[dates]


def _fill_proxy(df, col, dates, model=None):
    return _fill_with(_proxy_fill, df, col, dates, model)


def _fill_ml(df, col, dates, model=None):
    return _fill_with(_ml_fill, df, col, dates, model)


def _fill_with(fn, df, col, dates, model=None):
    masked = df.copy()
    masked.loc[dates, col] = np.nan
    vals = fn(masked, col, dates, model=model)
    return None if vals is None else pd.Series(vals, index=dates)


METHODS = {"carry_forward": _fill_carry,
           "interpolate": _fill_interpolate,
           "proxy_regression": _fill_proxy,
           "ml_random_forest": _fill_ml}

# methods that learn from history get one fit for all outages, not one per
# outage: same model applied everywhere, and far faster
FITTERS = {"proxy_regression": fit_proxy, "ml_random_forest": fit_ml}


def _blocks(index: pd.DatetimeIndex, hidden: pd.DatetimeIndex):
    """Split hidden dates into contiguous runs.

    The pipeline always repairs one continuous outage at a time. Handing a
    method every hidden day at once instead lets an anchored method drift
    across years of untouched data, which measures a mistake the pipeline
    would never make. So the harness reproduces the real call pattern.
    """
    pos = {d: i for i, d in enumerate(index)}
    runs, run = [], []
    for d in hidden:
        if run and pos[d] != pos[run[-1]] + 1:
            runs.append(pd.DatetimeIndex(run))
            run = []
        run.append(d)
    if run:
        runs.append(pd.DatetimeIndex(run))
    return runs


def _fill_in_blocks(fn, df, col, hidden, model=None):
    """Apply one method to each contiguous outage separately."""
    parts = []
    for block in _blocks(df.index, hidden):
        est = fn(df, col, block, model=model) if model is not None \
            else fn(df, col, block)
        if est is None:
            return None
        parts.append(est)
    return pd.concat(parts)


def mask_and_recover(df: pd.DataFrame, col: str, frac: float = 0.08,
                     seed: int = 7) -> pd.DataFrame:
    """Score every method on the same hidden points of one series."""
    hidden = _mask(df[col], frac, seed)
    truth = df.loc[hidden, col]
    true_ret = df[col].diff().dropna()
    rows = []
    for name, fn in METHODS.items():
        fitter = FITTERS.get(name)
        model = fitter(df, col, exclude=hidden) if fitter else None
        if fitter is not None and model is None:
            continue
        est = _fill_in_blocks(fn, df, col, hidden, model)
        if est is None:
            continue
        repaired = df[col].copy()
        repaired.loc[hidden] = est
        rep_ret = repaired.diff().loc[hidden].dropna()
        mae = float((est - truth).abs().mean())
        ks_p = float(stats.ks_2samp(
            rep_ret, true_ret.sample(min(len(true_ret), 250),
                                     random_state=0)).pvalue)
        true_vol = df[col].diff().loc[hidden].dropna().std()
        tail = float(rep_ret.std() / true_vol) if true_vol > 0 else np.nan
        rows.append({"method": name, "mae": round(mae, 4),
                     "ks_pvalue": round(ks_p, 4),
                     "tail_ratio": round(tail, 3),
                     "points_tested": len(hidden)})
    out = pd.DataFrame(rows).set_index("method")
    return out.sort_values("mae")
