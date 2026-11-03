"""Tests for the release gate - the logic that decides ship / no-ship.

Each of the three block conditions gets its own test, plus the
interaction that matters most: a run can be above threshold and still
be blocked by a critical failure or a regression. That is the property
that makes this a gate rather than a pass-rate display.
"""

from __future__ import annotations

from llmqa.gate import evaluate
from llmqa.runner import CaseResult, RunResult


def _run(*cases: CaseResult, provider="mock") -> RunResult:
    return RunResult(
        provider=provider,
        started_at="2026-01-01T00:00:00+00:00",
        duration_s=0.1,
        cases=list(cases),
    )


def _case(cid, passed, severity="major"):
    return CaseResult(
        case_id=cid, suite="s", severity=severity, passed=passed,
        checks=[] if passed else [{"check": "x", "passed": False, "detail": "d"}],
    )


def test_all_pass_ships():
    d = evaluate(_run(_case("a", True), _case("b", True)), threshold=0.9)
    assert d.passed
    assert d.exit_code == 0


def test_below_threshold_blocks():
    d = evaluate(_run(_case("a", True), _case("b", False)), threshold=0.9)
    assert not d.passed
    assert d.exit_code == 1
    assert any("below" in r for r in d.reasons)


def test_critical_failure_blocks_even_above_threshold():
    # 9 pass, 1 critical fail -> 90% pass rate, meets threshold, but blocks
    cases = [_case(f"p{i}", True) for i in range(9)]
    cases.append(_case("crit", False, severity="critical"))
    d = evaluate(_run(*cases), threshold=0.9)
    assert d.pass_rate == 0.9
    assert not d.passed
    assert any("critical" in r for r in d.reasons)


def test_regression_blocks_even_above_threshold():
    baseline = _run(_case("a", True), _case("b", True), _case("c", True))
    # same pass count, but 'a' broke and 'd'... no, keep count: a breaks, b fixed-stays
    current = _run(_case("a", False), _case("b", True), _case("c", True))
    d = evaluate(current, threshold=0.5, baseline=baseline)
    assert not d.passed
    assert len(d.regressions) == 1
    assert d.regressions[0]["case_id"] == "a"


def test_fix_is_reported_but_does_not_block():
    baseline = _run(_case("a", False), _case("b", True))
    current = _run(_case("a", True), _case("b", True))
    d = evaluate(current, threshold=0.5, baseline=baseline)
    assert d.passed
    assert len(d.fixes) == 1
    assert d.fixes[0]["case_id"] == "a"


def test_new_case_is_not_a_regression():
    baseline = _run(_case("a", True))
    current = _run(_case("a", True), _case("b", False))  # b is new and failing
    d = evaluate(current, threshold=0.0, baseline=baseline)
    assert d.regressions == []  # b did not regress; it never existed before


def test_allow_regressions_flag():
    baseline = _run(_case("a", True))
    current = _run(_case("a", False))
    blocked = evaluate(current, threshold=0.0, baseline=baseline)
    assert not blocked.passed
    allowed = evaluate(current, threshold=0.0, baseline=baseline, allow_regressions=True)
    assert allowed.passed  # regression reported but not blocking
    assert len(allowed.regressions) == 1


def test_empty_run_blocks():
    d = evaluate(_run(), threshold=0.9)
    assert not d.passed
    assert any("no cases" in r for r in d.reasons)
