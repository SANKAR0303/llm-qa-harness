"""The release gate.

This is the part that makes the project a QA tool rather than a
dashboard. A gate answers one question - does this model version ship -
and it answers in an exit code a CI system understands.

Three independent reasons to block, any one of which is enough:

  1. Pass rate below threshold.  Overall quality bar.
  2. Any critical case failed.   Some failures are not negotiable no
     matter how good the aggregate looks: a 98% pass rate is not
     reassuring if the 2% is "leaked a credential".
  3. Regression against baseline. A case that passed before and fails
     now is worse than one that never passed, because something
     changed. Aggregate pass rate hides this completely: swap five
     passes for five different passes and the number does not move.

Reason 3 is the one borrowed most directly from firmware release
gating, and it is the one most AI evaluation setups do not have.

Fixes - cases that failed in the baseline and pass now - are reported
too. They never block, but a release note that says what got better
is worth as much to a reviewer as one that says what broke.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .runner import RunResult


@dataclass
class GateDecision:
    """The verdict, with every reason recorded."""

    passed: bool
    pass_rate: float
    threshold: float
    reasons: list[str] = field(default_factory=list)
    regressions: list[dict[str, str]] = field(default_factory=list)
    fixes: list[dict[str, str]] = field(default_factory=list)
    critical_failures: list[dict[str, str]] = field(default_factory=list)
    baseline_provider: str | None = None

    @property
    def exit_code(self) -> int:
        return 0 if self.passed else 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "exit_code": self.exit_code,
            "pass_rate": round(self.pass_rate, 4),
            "threshold": self.threshold,
            "reasons": self.reasons,
            "regressions": self.regressions,
            "fixes": self.fixes,
            "critical_failures": self.critical_failures,
            "baseline_provider": self.baseline_provider,
        }


def load_baseline(path: str | Path) -> RunResult | None:
    """Load a previous results.json, or None if it is not usable.

    A missing baseline is normal - the first run has nothing to compare
    against. It must not be an error, or CI fails the day you add the
    gate.
    """
    p = Path(path)
    if not p.exists():
        return None
    try:
        return RunResult.from_dict(json.loads(p.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, OSError):
        return None


def evaluate(
    result: RunResult,
    threshold: float = 0.9,
    baseline: RunResult | None = None,
    allow_regressions: bool = False,
) -> GateDecision:
    """Decide whether this run may ship."""
    reasons: list[str] = []

    # --- 1. aggregate quality bar ---
    if result.total == 0:
        reasons.append("no cases were executed")
    elif result.pass_rate < threshold:
        reasons.append(
            f"pass rate {result.pass_rate:.1%} is below the "
            f"{threshold:.1%} threshold"
        )

    # --- 2. critical failures ---
    criticals = [
        {
            "case_id": c.case_id,
            "suite": c.suite,
            "reason": c.summary_line(),
        }
        for c in result.critical_failures()
    ]
    if criticals:
        ids = ", ".join(c["case_id"] for c in criticals)
        reasons.append(f"{len(criticals)} critical case(s) failed: {ids}")

    # --- 3. regressions against baseline ---
    regressions: list[dict[str, str]] = []
    fixes: list[dict[str, str]] = []
    baseline_provider: str | None = None

    if baseline is not None:
        baseline_provider = baseline.provider
        before = {c.case_id: c for c in baseline.cases}
        for case in result.cases:
            prior = before.get(case.case_id)
            if prior is None:
                continue  # new case: nothing to regress from
            if prior.passed and not case.passed:
                regressions.append(
                    {
                        "case_id": case.case_id,
                        "suite": case.suite,
                        "severity": case.severity,
                        "reason": case.summary_line(),
                    }
                )
            elif not prior.passed and case.passed:
                fixes.append(
                    {
                        "case_id": case.case_id,
                        "suite": case.suite,
                        "severity": case.severity,
                        "was": prior.summary_line(),
                    }
                )

        if regressions and not allow_regressions:
            ids = ", ".join(r["case_id"] for r in regressions)
            reasons.append(f"{len(regressions)} regression(s) vs baseline: {ids}")

    return GateDecision(
        passed=not reasons,
        pass_rate=result.pass_rate,
        threshold=threshold,
        reasons=reasons,
        regressions=regressions,
        fixes=fixes,
        critical_failures=criticals,
        baseline_provider=baseline_provider,
    )
