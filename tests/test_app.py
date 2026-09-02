"""Run the real Streamlit script headless and drive every widget.

This catches the class of bug unit tests miss: a tab that throws only when
rendered, a selectbox option that breaks one code path, a markdown string
that renders wrong. If this passes, every tab a viewer can click has
been executed.
"""
import warnings
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


@pytest.fixture(scope="module")
def app():
    warnings.filterwarnings("ignore")
    script = Path(__file__).resolve().parent.parent / "app.py"
    at = AppTest.from_file(str(script), default_timeout=180)
    at.run()
    return at


def test_app_runs_without_exception(app):
    assert not app.exception, [str(e.value) for e in app.exception]


def test_headline_metrics_present(app):
    labels = [m.label for m in app.metric]
    assert any("VaR with corrupted" in l for l in labels)
    assert any("Expected shortfall" in l for l in labels)


def test_every_selectbox_option_renders(app):
    for sb in app.selectbox:
        for opt in sb.options:
            sb.set_value(opt)
            app.run()
            assert not app.exception, (sb.label, opt,
                                       [str(e.value) for e in app.exception])


def test_dollar_amounts_are_not_mangled_into_math(app):
    """Streamlit treats $...$ in markdown as LaTeX; an unescaped pair turns
    '$2.62M. On the clean data it is $2.61M' into italic gibberish."""
    for md in app.markdown:
        body = md.value
        if "On the clean data it is" in body:
            assert "\\$" in body


def test_all_four_tabs_exist(app):
    assert len(app.tabs) == 4
