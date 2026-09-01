"""Synthetic market risk factor time series.

Six factors, ~6.5 years of business days, seeded so every run reproduces
the same history. A high volatility era is engineered in 2022 so the
stressed VaR window search has something real to find.
"""
import numpy as np
import pandas as pd

SEED = 42
START = "2020-01-02"
END = "2026-08-31"

# per factor: (start_level, mean_rev_level, mean_rev_speed, daily_vol)
RATE_PARAMS = {
    "usd2y": (1.50, 3.80, 0.004, 0.045),
    "usd5y": (1.70, 3.95, 0.004, 0.055),
    "usd10y": (1.90, 4.10, 0.004, 0.050),
}
STRESS_START, STRESS_END = "2022-02-01", "2022-11-30"
STRESS_VOL_MULT = 3.0


def business_days() -> pd.DatetimeIndex:
    return pd.bdate_range(START, END)


def _ou_path(rng, dates, start, mean, speed, vol, stress_mask):
    """Mean reverting path in level space, extra vol inside the stress era."""
    n = len(dates)
    x = np.empty(n)
    x[0] = start
    shocks = rng.standard_normal(n)
    for i in range(1, n):
        v = vol * (STRESS_VOL_MULT if stress_mask[i] else 1.0)
        x[i] = x[i - 1] + speed * (mean - x[i - 1]) + v * shocks[i]
    return x


def generate_market_data(seed: int = SEED) -> pd.DataFrame:
    """Return the clean golden copy history for all six factors."""
    rng = np.random.default_rng(seed)
    dates = business_days()
    stress = (dates >= STRESS_START) & (dates <= STRESS_END)

    df = pd.DataFrame(index=dates)
    for name, (s, m, sp, v) in RATE_PARAMS.items():
        df[name] = _ou_path(rng, dates, s, m, sp, v, stress)

    # swaption vol in vol points, lognormal-ish, spikes in the stress era
    vol_shocks = rng.standard_normal(len(dates)) * 0.015
    vol_shocks[stress] *= STRESS_VOL_MULT
    df["swaption_vol"] = 80.0 * np.exp(np.cumsum(vol_shocks - 0.5 * 0.015**2))
    df["swaption_vol"] = df["swaption_vol"].clip(lower=30.0)

    # eurusd spot, log returns
    fx_shocks = rng.standard_normal(len(dates)) * 0.005
    fx_shocks[stress] *= 1.8
    df["eurusd"] = 1.12 * np.exp(np.cumsum(fx_shocks))

    # ig credit spread in bp, mean reverting, widens in stress
    df["credit_spread"] = _ou_path(rng, dates, 130.0, 140.0, 0.01, 1.6, stress)
    df["credit_spread"] = df["credit_spread"].clip(lower=45.0)

    return df.round(6)


if __name__ == "__main__":
    d = generate_market_data()
    print(d.describe().T[["mean", "std", "min", "max"]])
    print(f"{len(d)} business days, {d.index[0].date()} to {d.index[-1].date()}")
