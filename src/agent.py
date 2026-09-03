"""Remediation report narrative: an LLM explains, deterministic code decides.

The LLM never touches data and never invents numbers. It gets the findings,
proposals and guardrail results as structured facts and writes the morning
data quality report. A number check guardrail verifies every figure in the
draft appears in the fact set; on any failure (or with no API key) we fall
back to the deterministic template. Same pattern as the trading engine:
the model proposes, controls dispose.
"""
import os
import re


def build_facts(findings, checks, var_corrupt, var_repaired,
                es_corrupt=None, es_repaired=None) -> dict:
    auto = [c for c in checks if c["accepted"] and not c["needs_review"]]
    # without a reviewer the pipeline treats routed repairs as approved
    review_ok = [c for c in checks if c["accepted"] and c["needs_review"]
                 and c.get("approved_at_review") is not False]
    review_no = [c for c in checks
                 if c["needs_review"] and c.get("approved_at_review") is False]
    rejected = [c for c in checks if not c["accepted"] and not c["needs_review"]]
    held = 0
    if len(findings) and "verdict" in findings:
        held = int((findings["verdict"] == "level break, review").sum())
    facts = {
        "n_findings": len(findings),
        "n_faults": len(auto) + len(review_ok) + len(rejected),
        "n_held": held,
        "n_accepted": len(auto) + len(review_ok),
        "n_auto": len(auto),
        "n_review_ok": len(review_ok),
        "n_review_no": len(review_no),
        "n_review": len(review_ok) + len(review_no),
        "n_rejected": len(rejected),
        "var_corrupt_m": round(var_corrupt / 1e6, 2),
        "var_repaired_m": round(var_repaired / 1e6, 2),
        "series_touched": sorted({c["series"] for c in auto + review_ok + rejected}),
    }
    if es_corrupt is not None and es_repaired is not None:
        facts["es_corrupt_m"] = round(es_corrupt / 1e6, 2)
        facts["es_repaired_m"] = round(es_repaired / 1e6, 2)
    return facts


def template_report(facts: dict) -> str:
    n_faults = facts.get("n_faults", facts["n_findings"])
    first = (f"Data quality report: {n_faults} faults found across "
             f"{len(facts['series_touched'])} series ({', '.join(facts['series_touched'])})")
    if facts.get("n_held"):
        first += f", plus {facts['n_held']} possible level shifts held for review"
    first += "."
    parts = []
    if "n_auto" in facts:
        parts.append(f"{facts['n_auto']} repaired automatically")
        if facts.get("n_review_ok") or facts.get("n_review_no"):
            parts.append(f"{facts['n_review_ok']} approved at review")
            parts.append(f"{facts['n_review_no']} rejected at review as real market moves")
        parts.append(f"{facts['n_rejected']} rejected by a guardrail")
        second = "; ".join(parts) + "."
    else:
        second = (f"{facts['n_accepted']} repairs auto-accepted by guardrails; "
                  + (f"{facts['n_rejected']} rejected; " if "n_rejected" in facts else "")
                  + f"{facts['n_review']} routed to human review on VaR materiality.")
    if "es_corrupt_m" in facts:
        third = (f"Expected shortfall moved from {facts['es_corrupt_m']}M on the "
                 f"corrupted inputs to {facts['es_repaired_m']}M after repair; "
                 f"99 pct VaR from {facts['var_corrupt_m']}M to {facts['var_repaired_m']}M.")
    else:
        third = (f"99 pct VaR moved from {facts['var_corrupt_m']}M (corrupted inputs) to "
                 f"{facts['var_repaired_m']}M after repair.")
    third += (" Every applied point is flagged and reversible; the golden copy "
              "was never edited in place.")
    return " ".join([first, second, third])


def numbers_check(text: str, facts: dict) -> bool:
    """Every number in the draft must exist in the fact set. No invented figures."""
    allowed = {float(v) for v in facts.values() if isinstance(v, (int, float))}
    allowed.add(float(len(facts["series_touched"])))
    allowed.add(99.0)
    # series names may contain digits (usd5y); they are labels, not figures
    for name in facts["series_touched"]:
        text = text.replace(name, "")
    found = re.findall(r"\d+(?:\.\d+)?", text)
    return all(float(n) in allowed for n in found)


def narrative(facts: dict) -> tuple:
    """Return (text, source). Tries the LLM, falls back to the template."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return template_report(facts), "template (no API key set)"
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        msg = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=400,
            messages=[{"role": "user", "content":
                       "Write a 3 sentence market data quality morning report "
                       "using ONLY these facts and ONLY these numbers, "
                       f"plain prose, no markdown: {facts}"}],
        )
        text = msg.content[0].text.strip()
        if numbers_check(text, facts):
            return text, "llm (passed number check)"
        return template_report(facts), "template (llm draft failed number check)"
    except Exception:
        return template_report(facts), "template (llm unavailable)"
