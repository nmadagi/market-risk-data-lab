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


def build_facts(findings, checks, var_corrupt, var_repaired) -> dict:
    accepted = [c for c in checks if c["accepted"]]
    review = [c for c in checks if c["needs_review"]]
    return {
        "n_findings": len(findings),
        "n_accepted": len(accepted),
        "n_review": len(review),
        "var_corrupt_m": round(var_corrupt / 1e6, 2),
        "var_repaired_m": round(var_repaired / 1e6, 2),
        "series_touched": sorted({c["series"] for c in checks}),
    }


def template_report(facts: dict) -> str:
    lines = [
        f"Data quality report: {facts['n_findings']} findings across "
        f"{len(facts['series_touched'])} series ({', '.join(facts['series_touched'])}).",
        f"{facts['n_accepted']} repairs auto-accepted by guardrails; "
        f"{facts['n_review']} routed to human review on VaR materiality.",
        f"99 pct VaR moved from {facts['var_corrupt_m']}M (corrupted inputs) to "
        f"{facts['var_repaired_m']}M after repair. All filled points are flagged "
        "and reversible; the golden copy was never edited in place.",
    ]
    return " ".join(lines)


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
