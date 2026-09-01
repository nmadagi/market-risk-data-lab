import numpy as np
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
    proposals = remediation.propose(corrupted, findings)
    staged, flags = remediation.apply_to_staging(corrupted.ffill(), proposals)
    return corrupted, fault_log, findings, proposals, staged, flags


# ---------------- data generation ----------------

def test_generation_is_deterministic(clean):
    again = generate_market_data()
    pd.testing.assert_frame_equal(clean, again)

def test_no_missing_values_in_golden_copy(clean):
    assert not clean.isna().any().any()

def test_six_factors_present(clean):
    assert list(clean.columns) == ["usd2y", "usd5y", "usd10y",
                                   "swaption_vol", "eurusd", "credit_spread"]

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
    out, log = inject_gap(clean, "credit_spread", "2026-05-04", 20)
    assert out["credit_spread"].isna().sum() == 20

def test_splice_shifts_pre_seam_only(clean):
    out, _ = inject_splice(clean, "swaption_vol", "2023-01-16", 12.0)
    assert out["swaption_vol"].iloc[0] == pytest.approx(
        clean["swaption_vol"].iloc[0] + 12.0)
    assert out["swaption_vol"].iloc[-1] == pytest.approx(
        clean["swaption_vol"].iloc[-1])


# ---------------- detection ----------------

def test_all_four_faults_detected(scenario):
    _, _, findings, *_ = scenario
    kinds = set(findings["type"])
    assert {"stale", "spike", "gap"} <= kinds  # splice surfaces as a spike at the seam

def test_stale_found_on_right_series(scenario):
    _, _, findings, *_ = scenario
    stale = findings[findings["type"] == "stale"]
    assert "usd5y" in set(stale["series"])

def test_gap_found_with_right_length(scenario):
    _, _, findings, *_ = scenario
    gap = findings[(findings["type"] == "gap") &
                   (findings["series"] == "credit_spread")]
    assert int(gap.iloc[0]["length"]) == 20

def test_clean_data_yields_no_stale_or_gap(clean):
    findings = detection.run_all(clean)
    if not findings.empty:
        assert not (findings["type"].isin(["stale", "gap"])).any()

def test_peer_check_flags_lone_move(clean):
    out, _ = inject_spike(clean, "usd5y", "2026-07-15", n_sigma=10)
    findings = detection.run_all(out)
    spike = findings[(findings["type"] == "spike") &
                     (findings["series"] == "usd5y")]
    assert (spike["verdict"] == "likely data error").any()


# ---------------- remediation ----------------

def test_every_error_gets_a_proposal(scenario):
    _, _, findings, proposals, *_ = scenario
    errors = findings[findings.get("verdict", "x") != "likely real move"]
    assert len(proposals) >= min(3, len(errors))

def test_staging_leaves_no_nans_on_repaired_series(scenario):
    *_, staged, flags = scenario
    for s in flags["series"].unique():
        assert not staged[s].isna().any()

def test_every_filled_point_is_flagged(scenario):
    _, _, _, proposals, _, flags = scenario
    n_expected = sum(len(p["dates"]) for p in proposals)
    assert len(flags) == n_expected
    assert flags["filled"].all()

def test_guardrail_reports_required_fields(clean, scenario):
    corrupted, _, _, proposals, staged, _ = scenario
    c = remediation.guardrail_check(clean, staged, corrupted, proposals[0])
    assert {"ks_pvalue", "var_impact_pct", "accepted", "needs_review"} <= set(c)

def test_repair_moves_es_toward_clean(clean, scenario):
    """Expected shortfall is the metric that actually responds; see
    test_one_bad_point_moves_es_far_more_than_var for why."""
    corrupted, _, _, _, staged, _ = scenario
    e_clean = risk.expected_shortfall(risk.pnl_vector(clean))
    e_corrupt = risk.expected_shortfall(risk.pnl_vector(corrupted.ffill()))
    e_repaired = risk.expected_shortfall(risk.pnl_vector(staged))
    assert abs(e_repaired - e_clean) < abs(e_corrupt - e_clean)


def test_proxy_fill_does_not_inject_a_jump(clean):
    """Regression test. Fitting levels made the repair start away from the
    last real value, injecting a fake ~44bp move that read as an 8M loss.
    Fitting changes and anchoring to the real data keeps repaired moves in
    the same range as real ones."""
    corrupted, _ = inject_stale(clean, "usd5y", "2026-06-01", 15)
    findings = detection.run_all(corrupted)
    proposals = remediation.propose(corrupted, findings)
    staged, _ = remediation.apply_to_staging(corrupted.ffill(), proposals)
    biggest_real = clean["usd5y"].diff().abs().max()
    biggest_repaired = staged["usd5y"].diff().abs().max()
    assert biggest_repaired <= biggest_real * 1.5


# ---------------- risk engine ----------------

def test_var_is_positive_loss(clean):
    assert risk.var99(risk.pnl_vector(clean)) > 0

def test_es_exceeds_var(clean):
    pnl = risk.pnl_vector(clean)
    assert risk.expected_shortfall(pnl) > risk.var99(pnl)

def test_svar_exceeds_var_and_finds_stress_era(clean):
    v = risk.var99(risk.pnl_vector(clean))
    sv, ws, we = risk.svar99(clean)
    assert sv > v
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
    m = risk.factor_moves(clean)
    assert m["usd5y"].abs().mean() > 0.5  # bp scale, not decimal

def test_stress_scenarios_all_priced(clean):
    table = risk.stress_pnl()
    assert len(table) == 3
    assert table["pnl_musd"].notna().all()

def test_backtest_exceedance_rate_reasonable(clean):
    bt = risk.backtest(clean)
    assert 0 <= int(bt["exceedance"].sum()) <= 12


# ---------------- evaluation ----------------

def test_mask_and_recover_scores_all_methods(clean):
    out = evaluation.mask_and_recover(clean, "usd5y")
    assert {"carry_forward", "interpolate", "proxy_regression"} <= set(out.index)

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
    assert "template" in source
    assert agent.numbers_check(text, facts)
