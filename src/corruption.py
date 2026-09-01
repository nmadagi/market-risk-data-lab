"""Fault injection: the four data failures every market data team fights.

Each injector returns a new DataFrame (no mutation) plus a fault log row,
so the app can show exactly what was done to the clean history.
"""
import pandas as pd


def inject_stale(df, col, start, days):
    """Feed stalls: the value freezes at the last good level."""
    out = df.copy()
    idx = out.loc[start:].index[:days]
    out.loc[idx, col] = out[col].loc[:start].iloc[-1]
    return out, _log("stale", col, idx[0], idx[-1], f"value frozen for {len(idx)} days")


def inject_spike(df, col, date, n_sigma=8.0):
    """One corrupt print, n sigma of the series' own daily vol."""
    out = df.copy()
    d = out.loc[date:].index[0]
    sigma = out[col].diff().std()
    out.loc[d, col] = out.loc[d, col] + n_sigma * sigma
    return out, _log("spike", col, d, d, f"{n_sigma:.0f} sigma jump on one print")


def inject_gap(df, col, start, days):
    """Missing values: feed outage or calendar mismatch."""
    out = df.copy()
    idx = out.loc[start:].index[:days]
    out.loc[idx, col] = float("nan")
    return out, _log("gap", col, idx[0], idx[-1], f"{len(idx)} missing days")


def inject_splice(df, col, date, shift):
    """Vendor switch: all history before the seam sits at a shifted level."""
    out = df.copy()
    seam = out.loc[date:].index[0]
    out.loc[:seam, col] = out.loc[:seam, col] + shift
    return out, _log("splice", col, out.index[0], seam, f"pre-seam level shifted {shift:+.2f}")


def _log(kind, col, start, end, detail):
    return {"fault": kind, "series": col, "start": start, "end": end, "detail": detail}


def apply_default_faults(df):
    """The demo scenario: four faults on three different series."""
    faults = []
    out, f = inject_stale(df, "usd5y", "2026-06-01", 15)
    faults.append(f)
    out, f = inject_spike(out, "eurusd", "2026-07-15")
    faults.append(f)
    out, f = inject_gap(out, "credit_spread", "2026-05-04", 20)
    faults.append(f)
    out, f = inject_splice(out, "swaption_vol", "2023-01-16", 12.0)
    faults.append(f)
    return out, pd.DataFrame(faults)
