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

# A yield curve is not three independent series. One common "level" factor
# drives most of the daily movement in every tenor, which is why adjacent
# curve points correlate around 0.85 to 0.97 in real markets. Each tenor
# loads on that common shock and keeps a small idiosyncratic part of its
# own, so the curve can still steepen and flatten.
CURVE_LOADINGS = {"usd2y": 0.88, "usd5y": 1.00, "usd10y": 0.94}
IDIO_SHARE = 0.32          # fraction of each tenor's vol that is its own
STRESS_START, STRESS_END = "2022-02-01", "2022-11-30"
STRESS_VOL_MULT = 3.0

# swaption vol: mean reverting in log space so it stays in a realistic band
# (roughly 70 to 165 vol points, spiking only in the stress era). A plain
# random walk drifts to nonsense over 6 years; implied vol is anchored.
VOL_MEAN_REV_SPEED = 0.03
VOL_LOG_DAILY_VOL = 0.015
VOL_CALM_LEVEL = 80.0      # long run implied vol
VOL_STRESS_LEVEL = 150.0   # implied vol spikes in level during a crisis,
                           # it does not merely wobble more


def business_days() -> pd.DatetimeIndex:
    return pd.bdate_range(START, END)


def _ou_path(rng, dates, start, mean, speed, vol, stress_mask, shocks=None,
             stress_mean=None):
    """Mean reverting path in level space, extra vol inside the stress era.

    Pass `shocks` to drive the path with a supplied shock series instead of
    fresh independent noise; that is how the rate tenors share a common
    curve factor. Pass `stress_mean` to pull the series toward a different
    level inside the stress era, which is how implied vol spikes rather
    than just becoming noisier.
    """
    n = len(dates)
    x = np.empty(n)
    x[0] = start
    if shocks is None:
        shocks = rng.standard_normal(n)
    for i in range(1, n):
        stressed = stress_mask[i]
        m = stress_mean if (stress_mean is not None and stressed) else mean
        v = vol * (STRESS_VOL_MULT if stressed else 1.0)
        x[i] = x[i - 1] + speed * (m - x[i - 1]) + v * shocks[i]
    return x


def _curve_shocks(rng, n):
    """One shared shock per day plus a small private shock per tenor.

    Scaled so each tenor's shock still has unit variance, which keeps the
    volatility of every series the same as before this common factor was
    introduced.
    """
    common = rng.standard_normal(n)
    out = {}
    for name, load in CURVE_LOADINGS.items():
        idio = rng.standard_normal(n)
        mixed = load * common + IDIO_SHARE * idio
        out[name] = mixed / np.sqrt(load ** 2 + IDIO_SHARE ** 2)
    return out


def generate_market_data(seed: int = SEED) -> pd.DataFrame:
    """Return the clean golden copy history for all six factors."""
    rng = np.random.default_rng(seed)
    dates = business_days()
    stress = (dates >= STRESS_START) & (dates <= STRESS_END)

    df = pd.DataFrame(index=dates)
    curve = _curve_shocks(rng, len(dates))
    for name, (s, m, sp, v) in RATE_PARAMS.items():
        df[name] = _ou_path(rng, dates, s, m, sp, v, stress, shocks=curve[name])

    # swaption vol in vol points. Mean reverting in log space: implied vol
    # is anchored, it spikes and decays back, it does not random walk to
    # nonsense levels over years. Long run level 80, stress era spikes.
    log_vol = _ou_path(rng, dates, np.log(VOL_CALM_LEVEL),
                       np.log(VOL_CALM_LEVEL), VOL_MEAN_REV_SPEED,
                       VOL_LOG_DAILY_VOL, stress,
                       stress_mean=np.log(VOL_STRESS_LEVEL))
    df["swaption_vol"] = np.clip(np.exp(log_vol), 30.0, 200.0)

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
