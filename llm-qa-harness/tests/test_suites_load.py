"""The shipped suites must load, validate and run under the mock provider.

This is the test that keeps the three domain suites honest: if someone
edits a suite and breaks the YAML, miscounts a check, or declares a
fixture that no longer satisfies its own checks, this fails in CI
rather than at demo time.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llmqa.judge import Judge
from llmqa.providers.mock import MockProvider
from llmqa.runner import Runner
from llmqa.suite import load_suites

SUITES_DIR = Path(__file__).resolve().parents[1] / "suites"


def test_suites_directory_exists():
    assert SUITES_DIR.is_dir(), "suites/ directory should ship with the project"


def test_all_suites_load():
    suites = load_suites([str(SUITES_DIR)])
    assert len(suites) >= 3
    names = {s.name for s in suites}
    assert {"telecom_knowledge", "safety_refusals", "prompt_injection_logs"} <= names


def test_every_case_has_at_least_one_check():
    for suite in load_suites([str(SUITES_DIR)]):
        for case in suite.cases:
            assert case.checks, f"{case.id} has no checks"


def test_shipped_fixtures_satisfy_their_own_checks():
    """Each mock_response should make its case pass.

    A suite whose own fixtures fail is a broken demo: the mock run is
    supposed to be green, because it is testing the harness, not a model.
    """
    suites = load_suites([str(SUITES_DIR)])
    # Mirror the CLI, which defaults the judge to the same provider. The
    # judge-graded case carries a mock_verdict fixture so this stays
    # deterministic and offline.
    provider = MockProvider()
    runner = Runner(provider, judge=Judge(provider), workers=1)
    result = runner.run(suites)
    failures = [c.case_id for c in result.cases if not c.passed]
    assert not failures, f"shipped fixtures should all pass, these did not: {failures}"


def test_critical_cases_exist_in_safety_suites():
    """The safety and injection suites must carry critical cases.

    If none are critical, the gate's critical-failure rule can never
    fire for them, which would defeat the purpose of those suites.
    """
    suites = {s.name: s for s in load_suites([str(SUITES_DIR)])}
    for name in ("safety_refusals", "prompt_injection_logs"):
        severities = {c.severity for c in suites[name].cases}
        assert "critical" in severities, f"{name} should have critical cases"


@pytest.mark.parametrize("name", ["telecom_knowledge", "safety_refusals",
                                  "prompt_injection_logs"])
def test_case_ids_unique_within_suite(name):
    suite = {s.name: s for s in load_suites([str(SUITES_DIR)])}[name]
    ids = [c.id for c in suite.cases]
    assert len(ids) == len(set(ids))
