"""Executes suites against a provider.

Three properties matter here and each one is a deliberate choice:

1. Parallel. Evaluation is almost entirely waiting on a network call,
   so a thread pool turns a 10-minute suite into a 1-minute suite.

2. Nothing aborts the run. A transport error, a malformed check, an
   exception inside a scorer - all become a failed case with a reason
   attached. A run that dies on case 7 of 60 tells you nothing about
   cases 8 to 60, and those are the ones you needed.

3. Results are plain data. RunResult serialises to JSON, which is what
   makes baselines, regression detection and CI artefacts possible.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from .judge import Judge
from .providers import CompletionRequest, Provider
from .scorers import CheckResult, ScorerError, run_pure_check
from .suite import Case, Suite


@dataclass
class CaseResult:
    """What happened to one case."""

    case_id: str
    suite: str
    severity: str
    passed: bool
    tags: list[str] = field(default_factory=list)
    checks: list[dict[str, Any]] = field(default_factory=list)
    response: str = ""
    latency_s: float = 0.0
    error: str | None = None
    attempts: int = 1

    @property
    def failed_checks(self) -> list[dict[str, Any]]:
        return [c for c in self.checks if not c["passed"]]

    def summary_line(self) -> str:
        """One-line reason this case failed, for reports."""
        if self.error:
            return f"error: {self.error}"
        reasons = [
            f"{c['check']}: {c['detail']}" if c["detail"] else c["check"]
            for c in self.failed_checks
        ]
        return "; ".join(reasons) or "unknown"


@dataclass
class RunResult:
    """Everything about one execution of one or more suites."""

    provider: str
    started_at: str
    duration_s: float
    cases: list[CaseResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.cases)

    @property
    def passed(self) -> int:
        return sum(1 for c in self.cases if c.passed)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def errored(self) -> int:
        return sum(1 for c in self.cases if c.error)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    def failures(self) -> list[CaseResult]:
        """Failed cases, most severe first, then by suite and id."""
        order = {"critical": 0, "major": 1, "minor": 2}
        return sorted(
            (c for c in self.cases if not c.passed),
            key=lambda c: (order.get(c.severity, 9), c.suite, c.case_id),
        )

    def critical_failures(self) -> list[CaseResult]:
        return [c for c in self.cases if not c.passed and c.severity == "critical"]

    def by_suite(self) -> dict[str, list[CaseResult]]:
        grouped: dict[str, list[CaseResult]] = {}
        for case in self.cases:
            grouped.setdefault(case.suite, []).append(case)
        return grouped

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "started_at": self.started_at,
            "duration_s": round(self.duration_s, 3),
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "errored": self.errored,
            "pass_rate": round(self.pass_rate, 4),
            "cases": [asdict(c) for c in self.cases],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunResult:
        """Rebuild from a results.json - used for baseline comparison."""
        result = cls(
            provider=data.get("provider", "unknown"),
            started_at=data.get("started_at", ""),
            duration_s=float(data.get("duration_s", 0.0)),
        )
        for raw in data.get("cases", []):
            result.cases.append(
                CaseResult(
                    case_id=raw["case_id"],
                    suite=raw.get("suite", ""),
                    severity=raw.get("severity", "major"),
                    passed=bool(raw.get("passed", False)),
                    tags=list(raw.get("tags", [])),
                    checks=list(raw.get("checks", [])),
                    response=raw.get("response", ""),
                    latency_s=float(raw.get("latency_s", 0.0)),
                    error=raw.get("error"),
                    attempts=int(raw.get("attempts", 1)),
                )
            )
        return result


class Runner:
    """Runs cases against a provider and scores the responses."""

    def __init__(
        self,
        provider: Provider,
        judge: Judge | None = None,
        workers: int = 4,
        max_response_chars: int = 4000,
    ) -> None:
        self.provider = provider
        self.judge = judge
        self.workers = max(1, workers)
        # Responses are stored in results.json; truncate so a chatty
        # model cannot produce a 40MB CI artefact.
        self.max_response_chars = max_response_chars

    # ---------------------------------------------------------------

    def _score(self, case: Case, response: str) -> list[CheckResult]:
        results: list[CheckResult] = []
        for check in case.checks:
            try:
                if check.type == "llm_judge":
                    if self.judge is None:
                        results.append(
                            CheckResult(
                                check="llm_judge",
                                passed=False,
                                detail=(
                                    "no judge configured; pass --judge <provider spec>"
                                ),
                            )
                        )
                    else:
                        results.append(
                            self.judge.grade(response, check.params, case.prompt)
                        )
                else:
                    results.append(run_pure_check(check.type, response, check.params))

            except ScorerError as exc:
                results.append(
                    CheckResult(check=check.type, passed=False, detail=str(exc))
                )
            except Exception as exc:  # noqa: BLE001 - a scorer bug is a failed case
                results.append(
                    CheckResult(
                        check=check.type,
                        passed=False,
                        detail=f"scorer raised {type(exc).__name__}: {exc}",
                    )
                )
        return results

    def run_case(self, case: Case) -> CaseResult:
        """Execute and score one case. Never raises."""
        started = time.perf_counter()
        try:
            completion = self.provider.complete(
                CompletionRequest(
                    prompt=case.prompt,
                    system=case.system,
                    case_id=case.id,
                    mock_response=case.mock_response,
                )
            )
        except Exception as exc:  # noqa: BLE001 - provider bug, not a crash
            return CaseResult(
                case_id=case.id,
                suite=case.suite,
                severity=case.severity,
                passed=False,
                tags=list(case.tags),
                latency_s=time.perf_counter() - started,
                error=f"provider raised {type(exc).__name__}: {exc}",
            )

        if not completion.ok:
            return CaseResult(
                case_id=case.id,
                suite=case.suite,
                severity=case.severity,
                passed=False,
                tags=list(case.tags),
                latency_s=completion.latency_s,
                error=completion.error,
                attempts=completion.attempts,
            )

        checks = self._score(case, completion.text)
        return CaseResult(
            case_id=case.id,
            suite=case.suite,
            severity=case.severity,
            passed=all(c.passed for c in checks),
            tags=list(case.tags),
            checks=[
                {"check": c.check, "passed": c.passed, "detail": c.detail}
                for c in checks
            ],
            response=completion.text[: self.max_response_chars],
            latency_s=completion.latency_s,
            attempts=completion.attempts,
        )

    def run(self, suites: list[Suite]) -> RunResult:
        """Execute every case in every suite."""
        cases = [case for suite in suites for case in suite.cases]
        started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        start = time.perf_counter()

        if self.workers == 1:
            results = [self.run_case(c) for c in cases]
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                results = list(pool.map(self.run_case, cases))

        return RunResult(
            provider=self.provider.describe(),
            started_at=started_at,
            duration_s=time.perf_counter() - start,
            cases=results,
        )
