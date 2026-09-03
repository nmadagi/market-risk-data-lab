"""A second opinion on anomalies from an Isolation Forest.

The rule based detectors in detection.py give a reason for every flag:
"14 identical prints", "move of 10.9 sigma, undone next day". An
Isolation Forest gives a score instead: how unusual is this day for this
series, judged on several features at once. Neither replaces the other.
The rules explain; the forest can catch combinations no single rule was
written for. Because the demo has an answer key (the fault log), the two
can be scored against each other on exactly the faults that were planted,
which is the only honest way to decide how much weight the model earns.

Features per (series, day), all scale free so six different markets can
share one model:
  abs_z       size of today's move in units of this series' recent volatility
  reversal    how much of today's move tomorrow undoes (1 = fully undone)
  peer_max_z  biggest same day move among correlated series
  zero_run    how many consecutive days the value has not changed
  missing     1 if the value is absent
"""
import numpy as np
import pandas as pd

from src.detection import PEERS, SPIKE_WARMUP

CONTAMINATION = 0.003   # about 30 flags across ~10,000 series-days
BUDGETS = (0.003, 0.01, 0.02)   # alert budgets for the trade-off table


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Long frame: one row per (date, series) with the five features."""
    frames = []
    for col in df.columns:
        s = df[col]
        d = s.diff()
        ewma_var = d.pow(2).ewm(alpha=0.06).mean().shift(1)
        z = d / np.sqrt(ewma_var)
        rev = (-d.shift(-1) / d).clip(-2, 2)
        zero_run = (d == 0).astype(int)
        zero_run = zero_run.groupby((zero_run == 0).cumsum()).cumsum()
        peers = [p for p in PEERS.get(col, []) if p in df.columns]
        if peers:
            pz = pd.concat([(df[p].diff() / df[p].diff().rolling(60).std().shift(1)).abs()
                            for p in peers], axis=1).max(axis=1)
        else:
            pz = pd.Series(0.0, index=df.index)
        f = pd.DataFrame({
            "abs_z": z.abs(), "reversal": rev, "peer_max_z": pz,
            "zero_run": zero_run.astype(float), "missing": s.isna().astype(float),
        }, index=df.index)
        f["series"] = col
        frames.append(f.iloc[SPIKE_WARMUP:])
    out = pd.concat(frames).reset_index().rename(columns={"index": "date"})
    return out.fillna({"abs_z": 0.0, "reversal": 0.0, "peer_max_z": 0.0})


FEATURES = ["abs_z", "reversal", "peer_max_z", "zero_run", "missing"]


def isolation_forest_flags(df: pd.DataFrame, contamination: float = CONTAMINATION,
                           feats: pd.DataFrame = None):
    """One Isolation Forest per series; return the flagged series-days.

    Per series rather than one joint model: each market has its own idea
    of normal, and a joint fit let the loudest series set the bar for all.
    Returns None if scikit-learn is not installed, so the app still runs.
    """
    try:
        from sklearn.ensemble import IsolationForest
    except ImportError:
        return None
    feats = build_features(df) if feats is None else feats
    parts = []
    for _, g in feats.groupby("series"):
        X = g[FEATURES].to_numpy()
        model = IsolationForest(n_estimators=200, contamination=contamination,
                                random_state=0).fit(X)
        g = g.copy()
        g["score"] = -model.score_samples(X)     # higher = more unusual
        g["flag"] = model.predict(X) == -1
        parts.append(g)
    out = pd.concat(parts)
    return out[out["flag"]].sort_values("score", ascending=False)


def budget_sweep(df: pd.DataFrame, fault_log: pd.DataFrame,
                 findings: pd.DataFrame) -> pd.DataFrame:
    """How many planted faults the forest finds as its alert budget grows,
    and what that budget costs in false alarms. The trade-off in one table."""
    feats = build_features(df)
    rows = []
    for cont in BUDGETS:
        flags = isolation_forest_flags(df, cont, feats)
        if flags is None:
            return pd.DataFrame()
        found = int(scorecard(fault_log, findings, flags)["isolation forest found it"].sum())
        fp = false_positives(fault_log, flags)
        rows.append({"days flagged": len(flags),
                     "planted faults found": f"{found} of {len(fault_log)}",
                     "false alarms": len(fp),
                     "false alarms in 2022 stress era": int((fp["date"].dt.year == 2022).sum())})
    return pd.DataFrame(rows)


def scorecard(fault_log: pd.DataFrame, findings: pd.DataFrame,
              ml_flags: pd.DataFrame) -> pd.DataFrame:
    """Did each planted fault get caught, by the rules and by the forest?"""
    rows = []
    for f in fault_log.itertuples():
        rules_hit = bool(((findings["series"] == f.series) &
                          (findings["start"] >= f.start - pd.Timedelta(days=1)) &
                          (findings["start"] <= f.end + pd.Timedelta(days=1))).any())
        if f.fault == "splice":
            # the seam is the only day a splice is visible as a move
            rules_hit = bool(((findings["series"] == f.series) &
                              (findings["start"] >= f.end) &
                              (findings["start"] <= f.end + pd.Timedelta(days=3))).any())
            ml_hit = bool(((ml_flags["series"] == f.series) &
                           (ml_flags["date"] >= f.end) &
                           (ml_flags["date"] <= f.end + pd.Timedelta(days=3))).any())
        else:
            ml_hit = bool(((ml_flags["series"] == f.series) &
                           (ml_flags["date"] >= f.start) &
                           (ml_flags["date"] <= f.end + pd.Timedelta(days=1))).any())
        rows.append({"planted fault": f.fault, "series": f.series,
                     "rules found it": rules_hit,
                     "isolation forest found it": ml_hit})
    return pd.DataFrame(rows)


def false_positives(fault_log: pd.DataFrame, ml_flags: pd.DataFrame) -> pd.DataFrame:
    """Forest flags that fall outside every planted fault window."""
    mask = pd.Series(True, index=ml_flags.index)
    for f in fault_log.itertuples():
        lo = f.end if f.fault == "splice" else f.start
        hi = f.end + pd.Timedelta(days=3)
        inside = (ml_flags["series"] == f.series) & (ml_flags["date"] >= lo) & (ml_flags["date"] <= hi)
        mask &= ~inside
    return ml_flags[mask]


# ---------------------------------------------------------------------------
# Supervised detector: the fault injector is the teacher.
#
# An Isolation Forest has no labels, so it can only rank "unusual". But the
# injector can manufacture unlimited LABELED faults in unlimited synthetic
# histories. Train a classifier on many worlds full of planted faults, then
# test it on the demo world it has never seen. Every day gets a class:
# normal, stale, spike, gap, or splice.
# ---------------------------------------------------------------------------
from data.generate import generate_market_data
from src.corruption import inject_gap, inject_spike, inject_splice, inject_stale

CLASSES = ["normal", "stale", "spike", "gap", "splice"]
TRAIN_SEEDS = tuple(range(1000, 1016))     # sixteen synthetic worlds
TEST_SEED = 42                             # the demo world, never trained on

# Faults planted per world. One of each was too few: with twelve examples of
# a one-day event the model never saw the natural spread of the reversal
# feature and hedged on anything slightly unusual. Several per world costs
# nothing (the slow part is generating the history) and multiplies the
# labeled examples of the rare classes.
PER_WORLD = {"stale": 6, "spike": 6, "gap": 6, "splice": 3}


def make_world(seed: int):
    """One synthetic history with many labeled faults planted in it.

    Splices are capped at one per series: each splice shifts all history
    before its seam, so two on the same series would compound into levels
    no market produces.
    """
    rng = np.random.default_rng(seed)
    clean = generate_market_data(seed)
    cols, dates = list(clean.columns), clean.index
    labels = pd.DataFrame("normal", index=dates, columns=cols)
    used = {c: np.zeros(len(dates), bool) for c in cols}
    spliced = set()
    out = clean
    lo, hi = 300, len(dates) - 40

    plan = [k for k, n in PER_WORLD.items() for _ in range(n)]
    rng.shuffle(plan)
    for kind in plan:
        for _ in range(30):                     # a few tries to find room
            col = cols[rng.integers(len(cols))]
            if kind == "splice" and col in spliced:
                continue
            n = 1 if kind in ("spike", "splice") else int(rng.integers(5, 25))
            i = int(rng.integers(lo, hi - n))
            if used[col][max(0, i - 3):i + n + 3].any():
                continue
            start = dates[i]
            if kind == "stale":
                out, _ = inject_stale(out, col, start, n + 1)
                labels.loc[dates[i + 1:i + 1 + n], col] = "stale"
            elif kind == "gap":
                out, _ = inject_gap(out, col, start, n)
                labels.loc[dates[i:i + n], col] = "gap"
            elif kind == "spike":
                out, _ = inject_spike(out, col, start, float(rng.uniform(4, 12)))
                labels.loc[dates[i:i + 1], col] = "spike"
            else:
                shift = float(rng.uniform(3, 9) * clean[col].diff().std()
                              * rng.choice([-1, 1]))
                out, _ = inject_splice(out, col, start, shift)
                labels.loc[dates[i + 1:i + 2], col] = "splice"
                spliced.add(col)
            used[col][max(0, i - 3):i + n + 3] = True
            break
    return out, labels


def labeled_features(seed: int) -> pd.DataFrame:
    corrupted, labels = make_world(seed)
    feats = build_features(corrupted)
    long = labels.rename_axis("date").reset_index().melt(
        id_vars="date", var_name="series", value_name="label")
    return feats.merge(long, on=["date", "series"], how="left")


def train_classifier(seeds=TRAIN_SEEDS):
    """Gradient boosting on every series-day of every training world."""
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
    except ImportError:
        return None
    data = pd.concat([labeled_features(s) for s in seeds], ignore_index=True)
    X = data[FEATURES].to_numpy()
    y = data["label"].to_numpy()
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                           class_weight="balanced",
                                           random_state=0)
    model.fit(X, y)
    return model


def classify(df: pd.DataFrame, model) -> pd.DataFrame:
    """Flag every series-day the model does not call normal."""
    feats = build_features(df)
    feats["predicted"] = model.predict(feats[FEATURES].to_numpy())
    proba = model.predict_proba(feats[FEATURES].to_numpy())
    feats["confidence"] = proba.max(axis=1).round(3)
    return feats[feats["predicted"] != "normal"].sort_values("date")


def classifier_scorecard(fault_log: pd.DataFrame, flagged: pd.DataFrame) -> pd.DataFrame:
    """For each planted fault: found, and found with the right name?"""
    rows = []
    for f in fault_log.itertuples():
        lo = f.end if f.fault == "splice" else f.start
        hi = f.end + pd.Timedelta(days=3)
        hits = flagged[(flagged["series"] == f.series) &
                       (flagged["date"] >= lo) & (flagged["date"] <= hi)]
        rows.append({"planted fault": f.fault, "series": f.series,
                     "model found it": len(hits) > 0,
                     "model named it correctly": bool((hits["predicted"] == f.fault).any()),
                     "days flagged in window": len(hits)})
    return pd.DataFrame(rows)


REPAIR_CONFIDENCE = 0.80   # below this the model is guessing; send it to a human


def findings_from_model(flagged: pd.DataFrame) -> pd.DataFrame:
    """Turn per-day predictions into events in the findings schema the
    remediation step consumes: one row per run of consecutive flagged days
    on one series with one predicted class.

    stale, gap and spike become repair candidates, but only when the model
    is confident. A low confidence call is the model guessing, and guessing
    is exactly when an automatic repair is most likely to erase a real
    market move, so those go to a human instead. A predicted splice is
    always held: a permanent shift cannot be told from a real repricing
    from inside the series.
    """
    from src.detection import VERDICT_BREAK, VERDICT_ERROR
    cols = ["type", "series", "start", "length", "detail", "verdict"]
    if flagged is None or flagged.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for (series, kind), g in flagged.sort_values("date").groupby(["series", "predicted"]):
        dates = list(g["date"])
        runs, run = [], [dates[0]]
        for prev, cur in zip(dates, dates[1:]):
            if (cur - prev).days <= 4:      # consecutive business days
                run.append(cur)
            else:
                runs.append(run); run = [cur]
        runs.append(run)
        for run in runs:
            conf = g[g["date"].isin(run)]["confidence"].mean()
            if kind == "splice" or conf < REPAIR_CONFIDENCE:
                why = ("possible level shift" if kind == "splice"
                       else f"unclear, model called it {kind}")
                rows.append({"type": "spike", "series": series, "start": run[0],
                             "length": len(run),
                             "detail": f"{why}, confidence {conf:.2f}",
                             "verdict": VERDICT_BREAK})
            elif kind == "spike":
                rows.append({"type": "spike", "series": series, "start": run[0],
                             "length": len(run),
                             "detail": f"bad print, confidence {conf:.2f}",
                             "verdict": VERDICT_ERROR})
            else:
                rows.append({"type": kind, "series": series, "start": run[0],
                             "length": len(run),
                             "detail": f"{len(run)} {'frozen' if kind == 'stale' else 'missing'} days, confidence {conf:.2f}",
                             "verdict": None})
    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values("start").reset_index(drop=True)
