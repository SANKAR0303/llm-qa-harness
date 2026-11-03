"""Reporters: the same run, written for three different readers.

  results.json  -> machines. The baseline for the next run reads this.
  JUnit XML     -> CI. GitHub Actions, Jenkins and friends render it
                   natively as a test report with per-case detail.
  Markdown      -> humans. Pasted into a PR comment or a release note.

Writing three formats from one result object is not duplication; it is
the thing that lets a gate decision be read by a pipeline, a reviewer
and a dashboard without any of them parsing each other's output.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom

from .gate import GateDecision
from .runner import RunResult

SEVERITY_ICON = {"critical": "[CRIT]", "major": "[MAJ] ", "minor": "[MIN] "}


# --------------------------------------------------------------------
# JSON
# --------------------------------------------------------------------


def write_json(result: RunResult, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return p


# --------------------------------------------------------------------
# JUnit XML
# --------------------------------------------------------------------


def _strip_control_chars(text: str) -> str:
    """XML 1.0 forbids most control characters; model output has them."""
    return "".join(
        ch for ch in text if ch in "\t\n\r" or (ord(ch) >= 32 and ord(ch) != 127)
    )


def write_junit(result: RunResult, path: str | Path) -> Path:
    """Write a JUnit XML report, one <testsuite> per suite."""
    root = ET.Element("testsuites")
    root.set("name", "llm-qa-harness")
    root.set("tests", str(result.total))
    root.set("failures", str(result.failed))
    root.set("errors", str(result.errored))
    root.set("time", f"{result.duration_s:.3f}")

    for suite_name, cases in result.by_suite().items():
        suite_el = ET.SubElement(root, "testsuite")
        suite_el.set("name", suite_name)
        suite_el.set("tests", str(len(cases)))
        suite_el.set("failures", str(sum(1 for c in cases if not c.passed and not c.error)))
        suite_el.set("errors", str(sum(1 for c in cases if c.error)))
        suite_el.set("time", f"{sum(c.latency_s for c in cases):.3f}")

        for case in cases:
            case_el = ET.SubElement(suite_el, "testcase")
            case_el.set("name", case.case_id)
            case_el.set("classname", f"{suite_name}.{case.severity}")
            case_el.set("time", f"{case.latency_s:.3f}")

            if case.error:
                err = ET.SubElement(case_el, "error")
                err.set("message", _strip_control_chars(case.error)[:200])
                err.set("type", "ProviderError")
                err.text = _strip_control_chars(case.error)
            elif not case.passed:
                fail = ET.SubElement(case_el, "failure")
                fail.set("message", _strip_control_chars(case.summary_line())[:200])
                fail.set("type", f"{case.severity}Failure")
                body = [f"severity: {case.severity}", "", "failed checks:"]
                for check in case.failed_checks:
                    body.append(f"  - {check['check']}: {check['detail']}")
                body += ["", "response:", case.response[:1500]]
                fail.text = _strip_control_chars("\n".join(body))

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    pretty = minidom.parseString(ET.tostring(root, encoding="unicode")).toprettyxml(
        indent="  "
    )
    p.write_text(pretty, encoding="utf-8")
    return p


# --------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------


def render_markdown(result: RunResult, decision: GateDecision | None = None) -> str:
    """Render a human-readable report."""
    lines: list[str] = ["# LLM QA Report", ""]

    if decision is not None:
        verdict = "PASS" if decision.passed else "FAIL"
        lines += [f"**Gate: {verdict}**", ""]
        if decision.reasons:
            lines.append("Blocked because:")
            lines += [f"- {r}" for r in decision.reasons]
            lines.append("")

    lines += [
        f"- Provider: `{result.provider}`",
        f"- Started: {result.started_at}",
        f"- Duration: {result.duration_s:.1f}s",
        f"- Result: **{result.passed}/{result.total} passed "
        f"({result.pass_rate:.1%})**",
    ]
    if result.errored:
        lines.append(f"- Errored: {result.errored} (counted as failures)")
    lines.append("")

    # --- per-suite table ---
    lines += [
        "## Suites",
        "",
        "| Suite | Passed | Total | Pass rate | Critical fails |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, cases in sorted(result.by_suite().items()):
        passed = sum(1 for c in cases if c.passed)
        crit = sum(1 for c in cases if not c.passed and c.severity == "critical")
        rate = passed / len(cases) if cases else 0.0
        lines.append(
            f"| {name} | {passed} | {len(cases)} | {rate:.0%} | {crit} |"
        )
    lines.append("")

    # --- failures, most severe first ---
    failures = result.failures()
    if failures:
        lines += ["## Failures", ""]
        for case in failures:
            icon = SEVERITY_ICON.get(case.severity, "")
            lines.append(f"### {icon} `{case.case_id}` ({case.suite})")
            lines.append("")
            lines.append(f"- Severity: **{case.severity}**")
            if case.tags:
                lines.append(f"- Tags: {', '.join(f'`{t}`' for t in case.tags)}")
            if case.error:
                lines.append(f"- Error: `{case.error}`")
            else:
                lines.append("- Failed checks:")
                for check in case.failed_checks:
                    detail = f" - {check['detail']}" if check["detail"] else ""
                    lines.append(f"  - `{check['check']}`{detail}")
                snippet = case.response.strip().replace("\n", " ")[:220]
                if snippet:
                    lines.append(f"- Response: `{snippet}`")
            lines.append("")
    else:
        lines += ["## Failures", "", "None.", ""]

    # --- movement against baseline ---
    if decision is not None and decision.baseline_provider is not None:
        lines += ["## Against baseline", ""]
        lines.append(f"Baseline provider: `{decision.baseline_provider}`")
        lines.append("")

        if decision.regressions:
            lines += [f"### Regressions ({len(decision.regressions)})", ""]
            lines.append("Passed before, failing now:")
            lines.append("")
            for reg in decision.regressions:
                lines.append(
                    f"- `{reg['case_id']}` ({reg['severity']}) - {reg['reason']}"
                )
            lines.append("")
        else:
            lines += ["### Regressions", "", "None.", ""]

        if decision.fixes:
            lines += [f"### Fixes ({len(decision.fixes)})", ""]
            lines.append("Failing before, passing now:")
            lines.append("")
            for fix in decision.fixes:
                lines.append(
                    f"- `{fix['case_id']}` ({fix['severity']}) - was: {fix['was']}"
                )
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_markdown(
    result: RunResult,
    path: str | Path,
    decision: GateDecision | None = None,
) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render_markdown(result, decision), encoding="utf-8")
    return p


# --------------------------------------------------------------------
# Terminal
# --------------------------------------------------------------------


def render_console(result: RunResult, decision: GateDecision | None = None) -> str:
    """Compact summary for stdout."""
    lines = [
        "",
        f"  provider   {result.provider}",
        f"  cases      {result.passed}/{result.total} passed "
        f"({result.pass_rate:.1%})",
        f"  duration   {result.duration_s:.1f}s",
    ]
    if result.errored:
        lines.append(f"  errored    {result.errored}")

    failures = result.failures()
    if failures:
        lines += ["", "  failures (most severe first):"]
        for case in failures[:15]:
            lines.append(
                f"    [{case.severity:<8}] {case.case_id:<28} {case.summary_line()[:70]}"
            )
        if len(failures) > 15:
            lines.append(f"    ... and {len(failures) - 15} more")

    if decision is not None:
        lines += ["", f"  gate       {'PASS' if decision.passed else 'FAIL'}"]
        for reason in decision.reasons:
            lines.append(f"             - {reason}")
        if decision.fixes:
            lines.append(f"  fixes      {len(decision.fixes)} case(s) now passing")

    lines.append("")
    return "\n".join(lines)
