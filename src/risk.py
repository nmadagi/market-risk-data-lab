"""The risk engine: sensitivity-based historical simulation.

Positions are held as sensitivities (the standard fast approximation):
DV01 per curve node in dollars per basis point, vega in dollars per vol
point, an FX notional, and a credit spread DV01. Each historical day's
factor moves are replayed against those sensitivities to get a simulated
P&L vector; VaR is a percentile of that vector.
"""
import numpy as np
import pandas as pd

# the demo book: long duration via 5y, a 2s5s steepener flavor, long vega,
# long eur, long credit risk (loses when spreads widen)
PORTFOLIO = {
    "usd2y": -60_000.0,        # dv01 $/bp (short the 2y)
    "usd5y": 180_000.0,        # dv01 $/bp (long the 5y)
    "usd10y": 40_000.0,        # dv01 $/bp
    "swaption_vol": 95_000.0,  # vega $/vol point
    "eurusd": 25_000_000.0,    # fx notional $
    "credit_spread": -30_000.0,  # spread dv01 $/bp
}

VAR_LOOKBACK = 500
SVAR_WINDOW = 250

STRESS_SCENARIOS = {
    "rapid tightening": {"usd2y": 60, "usd5y": 45, "usd10y": 35,
                         "swaption_vol": 25, "eurusd": -0.03, "credit_spread": 40},
    "flight to quality": {"usd2y": -40, "usd5y": -50, "usd10y": -55,
                          "swaption_vol": 35, "eurusd": -0.05, "credit_spread": 90},
    "vol shock only": {"usd2y": 0, "usd5y": 0, "usd10y": 0,
                       "swaption_vol": 45, "eurusd": 0, "credit_spread": 15},
}


def factor_moves(df: pd.DataFrame) -> pd.DataFrame:
    """Daily moves in the units sensitivities expect.

    Rates and spreads: absolute change in basis points (safe near zero).
    Vol: absolute change in vol points. FX: log return.
    """
    m = pd.DataFrame(index=df.index)
    for c in ("usd2y", "usd5y", "usd10y"):
        m[c] = df[c].diff() * 100.0
    m["credit_spread"] = df["credit_spread"].diff()
    m["swaption_vol"] = df["swaption_vol"].diff()
    m["eurusd"] = np.log(df["eurusd"]).diff()
    return m.dropna()


def pnl_vector(df: pd.DataFrame, portfolio: dict = None,
               lookback: int = VAR_LOOKBACK) -> pd.Series:
    """Replay the last `lookback` days of moves against the book."""
    p = portfolio or PORTFOLIO
    m = factor_moves(df).iloc[-lookback:]
    pnl = (
        -m["usd2y"] * p["usd2y"] - m["usd5y"] * p["usd5y"]
        - m["usd10y"] * p["usd10y"]
        + m["swaption_vol"] * p["swaption_vol"]
        + m["eurusd"] * p["eurusd"]
        - m["credit_spread"] * p["credit_spread"]
    )
    return pnl


def var99(pnl: pd.Series) -> float:
    """99 pct one day VaR, reported as a positive dollar loss."""
    return float(-np.percentile(pnl, 1))


def expected_shortfall(pnl: pd.Series, conf: float = 0.99) -> float:
    cut = np.percentile(pnl, (1 - conf) * 100)
    return float(-pnl[pnl <= cut].mean())


def svar99(df: pd.DataFrame, portfolio: dict = None) -> tuple:
    """Search every rolling window for the worst VaR on the CURRENT book.

    Returns (svar, window_start, window_end). This is the Basel 2.5 idea:
    the stressed window is chosen for the portfolio you hold today.
    """
    p = portfolio or PORTFOLIO
    m = factor_moves(df)
    worst, w_start, w_end = 0.0, None, None
    step = 21  # monthly steps keep the search fast and the answer stable
    for i in range(0, len(m) - SVAR_WINDOW, step):
        w = m.iloc[i:i + SVAR_WINDOW]
        pnl = (
            -w["usd2y"] * p["usd2y"] - w["usd5y"] * p["usd5y"]
            - w["usd10y"] * p["usd10y"]
            + w["swaption_vol"] * p["swaption_vol"]
            + w["eurusd"] * p["eurusd"]
            - w["credit_spread"] * p["credit_spread"]
        )
        v = var99(pnl)
        if v > worst:
            worst, w_start, w_end = v, w.index[0], w.index[-1]
    return worst, w_start, w_end


def sensitivities_table(portfolio: dict = None) -> pd.DataFrame:
    p = portfolio or PORTFOLIO
    rows = [
        ("DV01 2y", p["usd2y"], "$ per bp"),
        ("DV01 5y", p["usd5y"], "$ per bp"),
        ("DV01 10y", p["usd10y"], "$ per bp"),
        ("Vega", p["swaption_vol"], "$ per vol point"),
        ("FX delta", p["eurusd"], "$ notional"),
        ("Spread DV01", p["credit_spread"], "$ per bp"),
    ]
    return pd.DataFrame(rows, columns=["sensitivity", "value", "unit"])


def stress_pnl(portfolio: dict = None) -> pd.DataFrame:
    """Apply each named scenario's coherent shocks to the book."""
    p = portfolio or PORTFOLIO
    rows = []
    for name, s in STRESS_SCENARIOS.items():
        pnl = (
            -s["usd2y"] * p["usd2y"] - s["usd5y"] * p["usd5y"]
            - s["usd10y"] * p["usd10y"]
            + s["swaption_vol"] * p["swaption_vol"]
            + s["eurusd"] * p["eurusd"]
            - s["credit_spread"] * p["credit_spread"]
        )
        rows.append({"scenario": name, "pnl_musd": round(pnl / 1e6, 2)})
    return pd.DataFrame(rows)


def backtest(df: pd.DataFrame, portfolio: dict = None,
             days: int = 250) -> pd.DataFrame:
    """Compare each day's realized P&L against the prior day's VaR."""
    pnl_all = pnl_vector(df, portfolio, lookback=len(df) - 1)
    records = []
    for i in range(len(pnl_all) - days, len(pnl_all)):
        history = pnl_all.iloc[max(0, i - VAR_LOOKBACK):i]
        if len(history) < 100:
            continue
        v = var99(history)
        realized = pnl_all.iloc[i]
        records.append({"date": pnl_all.index[i], "pnl": realized,
                        "var": -v, "exceedance": realized < -v})
    return pd.DataFrame(records).set_index("date")
