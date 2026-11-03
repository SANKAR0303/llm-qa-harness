"""Test suites as data.

A suite is a YAML file. Nothing in this module knows how to call a model
or how to score a response - it only turns text on disk into validated
objects, and fails loudly when a suite is malformed.

Keeping this separate matters: a broken suite file should be a clear
error at load time, not a confusing failure halfway through a run that
looks like a model regression.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

SEVERITIES = ("critical", "major", "minor")


class SuiteError(ValueError):
    """A suite file is malformed. Raised at load time, never at run time."""


@dataclass(frozen=True)
class Check:
    """One assertion against a model response."""

    type: str
    params: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        if not self.params:
            return self.type
        bits = ", ".join(f"{k}={v!r}" for k, v in sorted(self.params.items()))
        return f"{self.type}({bits})"


@dataclass(frozen=True)
class Case:
    """A single test case: one prompt, one response, many checks.

    All checks must pass for the case to pass. There is no partial
    credit - a release gate needs a binary answer per case.
    """

    id: str
    prompt: str
    checks: tuple[Check, ...]
    system: str | None = None
    severity: str = "major"
    tags: tuple[str, ...] = ()
    mock_response: str | None = None
    suite: str = ""

    @property
    def is_critical(self) -> bool:
        return self.severity == "critical"


@dataclass(frozen=True)
class Suite:
    """A named group of cases, loaded from one YAML file."""

    name: str
    description: str
    cases: tuple[Case, ...]
    path: str = ""
    system: str | None = None

    def filter_tags(self, include: set[str], exclude: set[str]) -> Suite:
        """Return a copy keeping only cases that match the tag filters.

        A case is kept when it carries at least one included tag (or no
        include filter was given) and none of the excluded tags.
        """
        kept = []
        for case in self.cases:
            tags = set(case.tags)
            if include and not (tags & include):
                continue
            if exclude and (tags & exclude):
                continue
            kept.append(case)
        return Suite(
            name=self.name,
            description=self.description,
            cases=tuple(kept),
            path=self.path,
            system=self.system,
        )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SuiteError(message)


def _parse_check(raw: Any, case_id: str, index: int) -> Check:
    where = f"case {case_id!r}, check #{index + 1}"
    _require(isinstance(raw, dict), f"{where}: expected a mapping")
    params = dict(raw)
    check_type = params.pop("type", None)
    _require(
        isinstance(check_type, str) and check_type.strip() != "",
        f"{where}: missing 'type'",
    )
    return Check(type=str(check_type), params=params)


def _parse_case(raw: Any, index: int, suite_name: str, suite_system: str | None) -> Case:
    where = f"{suite_name}: case #{index + 1}"
    _require(isinstance(raw, dict), f"{where}: expected a mapping")

    case_id = raw.get("id")
    _require(
        isinstance(case_id, str) and case_id.strip() != "",
        f"{where}: missing 'id'",
    )

    prompt = raw.get("prompt")
    _require(
        isinstance(prompt, str) and prompt.strip() != "",
        f"case {case_id!r}: missing 'prompt'",
    )

    severity = str(raw.get("severity", "major")).lower()
    _require(
        severity in SEVERITIES,
        f"case {case_id!r}: severity must be one of {SEVERITIES}, got {severity!r}",
    )

    raw_checks = raw.get("checks")
    _require(
        isinstance(raw_checks, list) and len(raw_checks) > 0,
        f"case {case_id!r}: needs at least one check",
    )
    checks = tuple(
        _parse_check(c, str(case_id), i) for i, c in enumerate(raw_checks)
    )

    raw_tags = raw.get("tags", []) or []
    _require(isinstance(raw_tags, list), f"case {case_id!r}: 'tags' must be a list")

    # A case-level system prompt wins over the suite-level default.
    system = raw.get("system", suite_system)

    return Case(
        id=str(case_id),
        prompt=str(prompt),
        checks=checks,
        system=str(system) if system is not None else None,
        severity=severity,
        tags=tuple(str(t) for t in raw_tags),
        mock_response=(
            str(raw["mock_response"]) if raw.get("mock_response") is not None else None
        ),
        suite=suite_name,
    )


def load_suite(path: str | Path) -> Suite:
    """Load and validate one suite file."""
    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise SuiteError(f"{path}: invalid YAML: {exc}") from exc
    except OSError as exc:
        raise SuiteError(f"{path}: cannot read file: {exc}") from exc

    _require(isinstance(data, dict), f"{path}: expected a mapping at the top level")

    name = str(data.get("name") or path.stem)
    suite_system = data.get("system")
    raw_cases = data.get("cases")
    _require(
        isinstance(raw_cases, list) and len(raw_cases) > 0,
        f"{path}: suite has no cases",
    )

    cases = tuple(
        _parse_case(
            c,
            i,
            name,
            str(suite_system) if suite_system is not None else None,
        )
        for i, c in enumerate(raw_cases)
    )

    seen: set[str] = set()
    for case in cases:
        _require(
            case.id not in seen,
            f"{path}: duplicate case id {case.id!r}",
        )
        seen.add(case.id)

    return Suite(
        name=name,
        description=str(data.get("description", "")),
        cases=cases,
        path=str(path),
        system=str(suite_system) if suite_system is not None else None,
    )


def load_suites(paths: list[str]) -> list[Suite]:
    """Load several suite files, expanding directories to their .yaml files."""
    resolved: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            resolved.extend(sorted(p.glob("*.yaml")) + sorted(p.glob("*.yml")))
        else:
            resolved.append(p)

    if not resolved:
        raise SuiteError(f"no suite files found in {paths}")

    return [load_suite(p) for p in resolved]
