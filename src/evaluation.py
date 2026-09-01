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
ship the simple one.
"""
import numpy as np
import pandas as pd
from scipy import stats

from src.detection import PEERS
from src.remediation import _proxy_fill


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


def _fill_proxy(df, col, dates):
    masked = df.copy()
    masked.loc[dates, col] = np.nan
    vals = _proxy_fill(masked, col, dates)
    if vals is None:
        return None
    return pd.Series(vals, index=dates)


METHODS = {"carry_forward": _fill_carry,
           "interpolate": _fill_interpolate,
           "proxy_regression": _fill_proxy}


def mask_and_recover(df: pd.DataFrame, col: str, frac: float = 0.08,
                     seed: int = 7) -> pd.DataFrame:
    """Score every method on the same hidden points of one series."""
    hidden = _mask(df[col], frac, seed)
    truth = df.loc[hidden, col]
    true_ret = df[col].diff().dropna()
    rows = []
    for name, fn in METHODS.items():
        est = fn(df, col, hidden)
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
