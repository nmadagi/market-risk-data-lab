import pandas as pd
import pytest

from data.generate import generate_market_data
from src import agent, detection, evaluation, remediation, risk
from src.corruption import (apply_default_faults, inject_gap, inject_spike,
                            inject_stale, inject_splice)


@pytest.fixture(scope="module")
def clean():
    return generate_market_data()


@pytest.fixture(scope="module")
def scenario(clean):
    corrupted, fault_log = apply_default_faults(clean)
    findings = detection.run_all(corrupted)
    staged, flags, proposals, checks, unresolved = remediation.run(
        corrupted, findings)
    return (corrupted, fault_log, findings, proposals, staged, flags, checks,
            unresolved)


# ---------------- data generation ----------------

def test_generation_is_deterministic(clean):
    pd.testing.assert_frame_equal(clean, generate_market_data())

def test_no_missing_values_in_golden_copy(clean):
    assert not clean.isna().any().any()

def test_six_factors_present(clean):
    assert list(clean.columns) == ["usd2y", "usd5y", "usd10y",
                                   "swaption_vol", "eurusd", "credit_spread"]

def test_business_days_only(clean):
    assert (clean.index.dayofweek < 5).all()

def test_stress_era_is_more_volatile(clean):
    stress = clean.loc["2022-02-01":"2022-11-30", "usd5y"].diff().std()
    calm = clean.loc["2024-01-01":"2024-12-31", "usd5y"].diff().std()
    assert stress > 2 * calm

def test_levels_are_sane(clean):
    assert clean["eurusd"].between(0.5, 2.0).all()
    assert clean["credit_spread"].ge(45).all()
    rates = clean[["usd2y", "usd5y", "usd10y"]]
    assert ((rates > -1.0) & (rates < 12.0)).all().all()

def test_implied_vol_stays_in_a_realistic_band(clean):
    """Regression test. A plain random walk drifted vol past 1000 points
    over six years, which is not a number any market has produced. Implied
    vol mean reverts, so the generator does too."""
    v = clean["swaption_vol"]
    assert v.between(40, 180).all()
    assert v.max() < 200  # never pinned against the clip

def test_stress_era_shows_up_in_vol(clean):
    stress_max = clean.loc["2022-02-01":"2022-11-30", "swaption_vol"].max()
    calm_max = clean.loc["2024", "swaption_vol"].max()
    assert stress_max > calm_max * 1.3


# ---------------- corruption ----------------

def test_injectors_do_not_mutate_input(clean):
    before = clean.copy()
    inject_stale(clean, "usd5y", "2026-06-01", 15)
    inject_spike(clean, "eurusd", "2026-07-15")
    inject_gap(clean, "credit_spread", "2026-05-04", 20)
    inject_splice(clean, "swaption_vol", "2023-01-16", 12.0)
    pd.testing.assert_frame_equal(clean, before)

def test_stale_freezes_value(clean):
    out, log = inject_stale(clean, "usd5y", "2026-06-01", 15)
    frozen = out.loc[log["start"]:log["end"], "usd5y"]
    assert frozen.nunique() == 1

def test_gap_creates_nans(clean):
    out, _ = inject_gap(clean, "credit_spread", "2026-05-04", 20)
    assert out["credit_spread"].isna().sum() == 20

def test_splice_shifts_pre_seam_only(clean):
    out, _ = inject_splice(clean, "swaption_vol", "2023-01-16", 12.0)
    assert out["swaption_vol"].iloc[0] == pytest.approx(
        clean["swaption_vol"].iloc[0] + 12.0)
    assert out["swaption_vol"].iloc[-1] == pytest.approx(
        clean["swaption_vol"].iloc[-1])

def test_fault_log_has_four_distinct_faults(scenario):
    _, fault_log, *_ = scenario
    assert sorted(fault_log["fault"]) == ["gap", "spike", "splice", "stale"]


# ---------------- detection ----------------

def test_stale_gap_and_spike_all_detected(scenario):
    _, _, findings, *_ = scenario
    assert {"stale", "spike", "gap"} <= set(findings["type"])

def test_stale_found_on_right_series(scenario):
    _, _, findings, *_ = scenario
    assert "usd5y" in set(findings[findings["type"] == "stale"]["series"])

def test_gap_found_with_right_length(scenario):
    _, _, findings, *_ = scenario
    gap = findings[(findings["type"] == "gap") &
                   (findings["series"] == "credit_spread")]
    assert int(gap.iloc[0]["length"]) == 20

def test_clean_data_yields_no_stale_or_gap(clean):
    findings = detection.run_all(clean)
    if not findings.empty:
        assert not findings["type"].isin(["stale", "gap"]).any()

def test_no_spikes_flagged_during_warmup(clean):
    """Regression test. With no history the EWMA vol estimate is noise and
    the first days of 2020 were flagged as 28 sigma spikes."""
    findings = detection.run_all(clean)
    spikes = findings[findings["type"] == "spike"] if not findings.empty else findings
    cutoff = clean.index[detection.SPIKE_WARMUP]
    assert (spikes["start"] >= cutoff).all() if len(spikes) else True

def test_injected_spike_reverses_and_is_called_an_error(clean):
    out, _ = inject_spike(clean, "usd5y", "2026-07-15", n_sigma=10)
    findings = detection.run_all(out)
    row = findings[(findings["type"] == "spike") &
                   (findings["series"] == "usd5y") &
                   (findings["start"] == pd.Timestamp("2026-07-15"))].iloc[0]
    assert row["reverses"] and row["verdict"] == detection.VERDICT_ERROR

def test_splice_seam_is_a_level_break_not_an_error(scenario):
    """A vendor level shift does not revert next day, so it must be held
    for a human, not interpolated away."""
    _, _, findings, *_ = scenario
    seam = findings[(findings["series"] == "swaption_vol") &
                    (findings["start"] == pd.Timestamp("2023-01-17"))]
    assert len(seam) == 1
    assert seam.iloc[0]["verdict"] == detection.VERDICT_BREAK

def test_peer_confirmed_move_is_called_real(clean):
    out = clean.copy()
    for col in ("usd2y", "usd5y", "usd10y"):
        out, _ = inject_spike(out, col, "2026-07-15", n_sigma=8)
    findings = detection.run_all(out)
    row = findings[(findings["series"] == "usd5y") &
                   (findings["start"] == pd.Timestamp("2026-07-15"))].iloc[0]
    assert row["verdict"] == detection.VERDICT_REAL


# ---------------- remediation ----------------

def test_level_breaks_get_no_proposal(scenario):
    _, _, findings, proposals, *_ = scenario
    held = findings[findings["verdict"] == detection.VERDICT_BREAK]
    proposed = {(p["series"], p["dates"][0]) for p in proposals}
    for _, h in held.iterrows():
        assert (h["series"], h["start"]) not in proposed

def test_stale_and_injected_spike_get_proposals(scenario):
    _, _, _, proposals, *_ = scenario
    kinds = {(p["series"], p["type"]) for p in proposals}
    assert ("usd5y", "stale") in kinds and ("usd5y", "spike") in kinds

def test_only_accepted_proposals_are_applied(scenario):
    """Regression test. Rejected repairs used to land in staging anyway."""
    corrupted, _, _, proposals, staged, flags, checks, _ = scenario
    base = corrupted.ffill()
    for p, c in zip(proposals, checks):
        touched = (staged.loc[p["dates"], p["series"]]
                   != base.loc[p["dates"], p["series"]]).any()
        assert touched == c["accepted"]
    n_expected = sum(len(p["dates"]) for p, c in zip(proposals, checks)
                     if c["accepted"])
    assert len(flags) == n_expected and flags["filled"].all()

def test_guardrail_scores_each_proposal_alone(scenario):
    """Regression test. Every proposal used to show the same VaR impact
    because the check measured the whole staged frame."""
    _, _, _, _, _, _, checks, _ = scenario
    impacts = {c["var_impact_pct"] for c in checks}
    assert len(impacts) > 1

def test_guardrail_never_sees_clean_data(scenario):
    corrupted, _, _, proposals, *_ = scenario
    c = remediation.guardrail_check(corrupted.ffill(), proposals[0])
    assert {"ks_pvalue", "var_impact_pct", "accepted", "needs_review",
            "points", "type"} <= set(c)

def test_staging_has_no_nans(scenario):
    _, _, _, _, staged, _, _, _ = scenario
    assert not staged.isna().any().any()

def test_repair_moves_es_toward_clean(clean, scenario):
    """Expected shortfall is the metric that actually responds; see
    test_one_bad_point_moves_es_far_more_than_var for why."""
    corrupted, _, _, _, staged, _, _, _ = scenario
    e_clean = risk.expected_shortfall(risk.pnl_vector(clean))
    e_corrupt = risk.expected_shortfall(risk.pnl_vector(corrupted.ffill()))
    e_repaired = risk.expected_shortfall(risk.pnl_vector(staged))
    assert abs(e_repaired - e_clean) < abs(e_corrupt - e_clean)
    assert abs(e_repaired - e_clean) / e_clean < 0.10

def test_proxy_fill_does_not_inject_a_jump(clean):
    """Regression test. Fitting levels made the repair start away from the
    last real value, injecting a fake ~44bp move that read as an 8M loss.
    Fitting changes and anchoring to the real data keeps repaired moves in
    the same range as real ones."""
    corrupted, _ = inject_stale(clean, "usd5y", "2026-06-01", 15)
    findings = detection.run_all(corrupted)
    staged = remediation.run(corrupted, findings)[0]
    biggest_real = clean["usd5y"].diff().abs().max()
    assert staged["usd5y"].diff().abs().max() <= biggest_real * 1.5

def test_proxy_fill_lands_on_next_real_observation(clean):
    corrupted, _ = inject_stale(clean, "usd5y", "2026-06-01", 15)
    findings = detection.run_all(corrupted)
    staged = remediation.run(corrupted, findings)[0]
    after = clean.loc["2026-06-22":].index[0]
    step_into_real = abs(staged.loc[after, "usd5y"] - staged["usd5y"].shift(1)[after])
    assert step_into_real < clean["usd5y"].diff().abs().quantile(0.99)


# ---------------- risk engine ----------------

def test_var_is_positive_loss(clean):
    assert risk.var99(risk.pnl_vector(clean)) > 0

def test_es_exceeds_var(clean):
    pnl = risk.pnl_vector(clean)
    assert risk.expected_shortfall(pnl) > risk.var99(pnl)

def test_svar_exceeds_var_and_finds_stress_era(clean):
    v = risk.var99(risk.pnl_vector(clean))
    sv, ws, we = risk.svar99(clean)
    assert sv > 2 * v
    assert ws.year <= 2022 <= we.year

def test_single_stale_factor_barely_moves_var(clean):
    """The uncomfortable finding this app exists to show: 99% VaR is the
    5th worst of 500 days, so losing 15 mid-range days to a stalled feed
    moves it almost not at all. The risk number is not a data alarm."""
    out, _ = inject_stale(clean, "usd5y", "2026-06-01", 15)
    v_clean = risk.var99(risk.pnl_vector(clean))
    v_stale = risk.var99(risk.pnl_vector(out))
    assert abs(v_stale - v_clean) / v_clean < 0.02

def test_one_bad_point_moves_es_far_more_than_var(clean):
    """One corrupt print becomes the single worst day. A percentile shifts
    by one rank and shrugs; an average over the tail absorbs the whole
    fake loss. This is why the VaR to expected shortfall shift under FRTB
    raises the stakes on data quality rather than lowering them."""
    out, _ = inject_spike(clean, "usd5y", "2026-07-15", n_sigma=8)
    p_clean, p_bad = risk.pnl_vector(clean), risk.pnl_vector(out)
    var_move = abs(risk.var99(p_bad) - risk.var99(p_clean)) / risk.var99(p_clean)
    es_move = abs(risk.expected_shortfall(p_bad) -
                  risk.expected_shortfall(p_clean)) / risk.expected_shortfall(p_clean)
    assert var_move < 0.02
    assert es_move > 0.25
    assert es_move > var_move * 10

def test_rate_moves_are_in_bp(clean):
    assert risk.factor_moves(clean)["usd5y"].abs().mean() > 0.5

def test_pnl_signs_match_positions():
    """Long duration loses when rates rise; long vega gains when vol rises."""
    idx = pd.bdate_range("2026-01-01", periods=3)
    df = pd.DataFrame({"usd2y": 3.0, "usd5y": [3.0, 3.1, 3.1], "usd10y": 3.0,
                       "swaption_vol": [80.0, 80.0, 85.0], "eurusd": 1.1,
                       "credit_spread": 120.0}, index=idx)
    pnl = risk.pnl_vector(df, lookback=2)
    assert pnl.iloc[0] < 0           # 5y up 10bp: long duration loses
    assert pnl.iloc[1] > 0           # vol up 5 points: long vega gains

def test_stress_scenarios_all_priced():
    table = risk.stress_pnl()
    assert len(table) == 3 and table["pnl_musd"].notna().all()

def test_backtest_exceedance_rate_reasonable(clean):
    bt = risk.backtest(clean)
    assert 0 <= int(bt["exceedance"].sum()) <= 12


# ---------------- evaluation ----------------

def test_mask_and_recover_scores_all_methods(clean):
    out = evaluation.mask_and_recover(clean, "usd5y")
    assert {"carry_forward", "interpolate", "proxy_regression"} <= set(out.index)

def test_series_without_peers_has_no_proxy_row(clean):
    out = evaluation.mask_and_recover(clean, "eurusd")
    assert "proxy_regression" not in out.index
    assert {"carry_forward", "interpolate"} <= set(out.index)

def test_carry_forward_smooths_the_tail(clean):
    out = evaluation.mask_and_recover(clean, "usd5y")
    assert out.loc["carry_forward", "tail_ratio"] < \
        out.loc["proxy_regression", "tail_ratio"]

def test_interpolation_beats_carry_on_accuracy(clean):
    out = evaluation.mask_and_recover(clean, "usd5y")
    assert out.loc["interpolate", "mae"] <= out.loc["carry_forward", "mae"]


# ---------------- agent guardrail ----------------

def test_template_report_passes_number_check():
    facts = {"n_findings": 5, "n_accepted": 3, "n_review": 1,
             "var_corrupt_m": 3.1, "var_repaired_m": 4.2,
             "series_touched": ["usd5y", "eurusd"]}
    assert agent.numbers_check(agent.template_report(facts), facts)

def test_invented_number_fails_check():
    facts = {"n_findings": 5, "n_accepted": 3, "n_review": 1,
             "var_corrupt_m": 3.1, "var_repaired_m": 4.2,
             "series_touched": ["usd5y"]}
    assert not agent.numbers_check("We found 5 findings and VaR was 9.9M", facts)

def test_narrative_falls_back_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    facts = {"n_findings": 2, "n_accepted": 2, "n_review": 0,
             "var_corrupt_m": 3.0, "var_repaired_m": 3.5,
             "series_touched": ["usd5y"]}
    text, source = agent.narrative(facts)
    assert "template" in source and agent.numbers_check(text, facts)

def test_build_facts_matches_scenario(scenario):
    _, _, findings, _, _, _, checks, _ = scenario
    facts = agent.build_facts(findings, checks, 2_620_000, 2_580_000)
    assert facts["n_findings"] == len(findings)
    assert facts["n_accepted"] == sum(c["accepted"] for c in checks)


# ---------------- data snapshot ----------------

def test_csv_snapshot_matches_generator(clean):
    """The CSVs in data/ are a snapshot of the generator. If someone edits
    the generator and forgets to re-export, this fails."""
    from data import export
    if not export.FILES["clean"].exists():
        pytest.skip("run python -m data.export first")
    on_disk = export.load_clean()
    pd.testing.assert_frame_equal(on_disk, clean.round(6), check_freq=False,
                                  check_names=False, atol=1e-6)


# ---------------- unresolved points ----------------

def test_rejected_gap_is_reported_as_unresolved(scenario):
    """Regression test. A rejected repair left 20 days silently carried
    forward with no audit trail. Carry forward has zero volatility, so
    that is the worst possible value to have in the data unannounced."""
    corrupted, _, _, _, _, _, _, unresolved = scenario
    gap = corrupted.index[corrupted["credit_spread"].isna()]
    reported = unresolved[unresolved["series"] == "credit_spread"]["date"]
    assert set(gap) == set(reported)
    assert reported.size == 20

def test_no_faulty_point_is_both_repaired_and_unresolved(scenario):
    _, _, _, _, _, flags, _, unresolved = scenario
    applied = set(zip(flags["series"], flags["date"]))
    listed = set(zip(unresolved["series"], unresolved["date"]))
    assert not (applied & listed)

def test_every_unresolved_row_says_why(scenario):
    *_, unresolved = scenario
    assert unresolved["reason"].str.len().gt(10).all()
    assert unresolved["value_in_use"].notna().all()


# ---------------- ml benchmark ----------------

def test_random_forest_is_benchmarked_where_peers_exist(clean):
    out = evaluation.mask_and_recover(clean, "usd5y")
    assert "ml_random_forest" in out.index

def test_random_forest_preserves_the_tail_better_than_interpolation(clean):
    """The finding: ranking by average accuracy ships interpolation,
    ranking by tail preservation ships the forest. For risk data the
    second criterion is the one that matters."""
    for col in ("usd5y", "usd10y", "swaption_vol"):
        out = evaluation.mask_and_recover(clean, col)
        assert out.loc["ml_random_forest", "tail_ratio"] > \
            out.loc["interpolate", "tail_ratio"]

def test_no_method_fully_preserves_the_tail(clean):
    """Honesty check: every fill flattens volatility somewhat, which is
    why filled points stay flagged."""
    out = evaluation.mask_and_recover(clean, "usd5y")
    assert (out["tail_ratio"] < 1.0).all()

def test_masked_blocks_are_filled_one_outage_at_a_time(clean):
    """Regression test. The harness used to hand every hidden day to a
    method at once, so an anchored method drifted across years and scored
    a mistake the pipeline would never make (tail ratio near 10)."""
    out = evaluation.mask_and_recover(clean, "usd5y")
    assert out["tail_ratio"].max() < 2.0
    assert out["mae"].max() < 0.5
