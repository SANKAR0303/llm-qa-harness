"""Tests for the runner and the provider layer.

The runner's contract is "never abort, turn every failure into a scored
case". The most important tests here are the ones that prove a broken
provider or a broken scorer produces a failed case rather than a crash.
"""

from __future__ import annotations

from llmqa.providers import CompletionRequest, CompletionResponse, Provider, parse_spec
from llmqa.providers.mock import MockProvider
from llmqa.runner import Runner
from llmqa.suite import Case, Check, Suite


def _suite(*cases: Case) -> Suite:
    return Suite(name="t", description="", cases=tuple(cases))


def _case(cid, prompt, checks, mock_response, severity="major"):
    return Case(
        id=cid,
        prompt=prompt,
        checks=tuple(checks),
        mock_response=mock_response,
        severity=severity,
        suite="t",
    )


def test_mock_serves_declared_fixture():
    p = MockProvider()
    resp = p.complete(CompletionRequest(prompt="x", mock_response="hello"))
    assert resp.ok
    assert resp.text == "hello"


def test_passing_case():
    case = _case(
        "c1", "prompt",
        [Check("contains_all", {"values": ["carrier"]})],
        "fault: no carrier",
    )
    result = Runner(MockProvider(), workers=1).run([_suite(case)])
    assert result.total == 1
    assert result.passed == 1
    assert result.pass_rate == 1.0


def test_all_checks_must_pass():
    case = _case(
        "c1", "prompt",
        [
            Check("contains_all", {"values": ["carrier"]}),
            Check("not_contains", {"values": ["carrier"]}),  # contradicts the first
        ],
        "fault: no carrier",
    )
    result = Runner(MockProvider(), workers=1).run([_suite(case)])
    assert result.passed == 0  # one check fails -> case fails


class _BrokenProvider(Provider):
    name = "broken"

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        raise RuntimeError("boom")


def test_provider_exception_becomes_failed_case_not_crash():
    case = _case("c1", "p", [Check("contains_all", {"values": ["x"]})], "x")
    result = Runner(_BrokenProvider(), workers=1).run([_suite(case)])
    assert result.total == 1
    assert result.failed == 1
    assert result.errored == 1
    assert "boom" in result.cases[0].error


class _ErrorProvider(Provider):
    name = "err"

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        return CompletionResponse(text="", latency_s=0.1, error="HTTP 500", attempts=4)


def test_transport_error_is_a_failure_with_reason():
    case = _case("c1", "p", [Check("contains_all", {"values": ["x"]})], "x")
    result = Runner(_ErrorProvider(), workers=1).run([_suite(case)])
    assert result.failed == 1
    assert result.cases[0].error == "HTTP 500"
    assert result.cases[0].attempts == 4


def test_misconfigured_check_fails_case_without_crashing():
    # numeric_close with no 'expected' -> ScorerError, caught as a failed check
    case = _case("c1", "p", [Check("numeric_close", {})], "320")
    result = Runner(MockProvider(), workers=1).run([_suite(case)])
    assert result.failed == 1
    assert not result.cases[0].passed


def test_failures_sorted_by_severity():
    cases = [
        _case("minor1", "p", [Check("contains_all", {"values": ["zzz"]})], "no", "minor"),
        _case("crit1", "p", [Check("contains_all", {"values": ["zzz"]})], "no", "critical"),
        _case("major1", "p", [Check("contains_all", {"values": ["zzz"]})], "no", "major"),
    ]
    result = Runner(MockProvider(), workers=1).run([_suite(*cases)])
    order = [c.severity for c in result.failures()]
    assert order == ["critical", "major", "minor"]


def test_parallel_and_serial_agree():
    cases = [
        _case(f"c{i}", "p", [Check("contains_all", {"values": ["ok"]})], "ok")
        for i in range(20)
    ]
    serial = Runner(MockProvider(), workers=1).run([_suite(*cases)])
    parallel = Runner(MockProvider(), workers=8).run([_suite(*cases)])
    assert serial.passed == parallel.passed == 20


def test_roundtrip_through_dict():
    case = _case("c1", "p", [Check("contains_all", {"values": ["ok"]})], "ok")
    result = Runner(MockProvider(), workers=1).run([_suite(case)])
    from llmqa.runner import RunResult

    restored = RunResult.from_dict(result.to_dict())
    assert restored.total == result.total
    assert restored.cases[0].case_id == "c1"


def test_spec_parsing_handles_ollama_tag_and_url():
    assert parse_spec("openai:qwen2.5:7b@http://localhost:11434/v1") == (
        "openai",
        "qwen2.5:7b",
        "http://localhost:11434/v1",
    )
    assert parse_spec("anthropic:claude-sonnet-4-5") == (
        "anthropic",
        "claude-sonnet-4-5",
        None,
    )
