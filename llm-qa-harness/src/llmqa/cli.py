"""Command-line entry point.

    llmqa run suites/*.yaml --provider mock
    llmqa run suites/ --provider anthropic:claude-sonnet-4-5 --baseline baseline.json
    llmqa gate reports/results.json --baseline baseline.json --min-pass-rate 0.95

The gate runs as part of `run`, so a single command is enough in CI.
`gate` exists separately so a pipeline can re-evaluate a stored result
against a different threshold without paying for the run again.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .gate import evaluate, load_baseline
from .judge import Judge
from .providers import ProviderError, build_provider
from .report import (
    render_console,
    write_json,
    write_junit,
    write_markdown,
)
from .runner import Runner, RunResult
from .suite import SuiteError, load_suites


def _add_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "suites",
        nargs="+",
        help="suite YAML files or directories containing them",
    )
    p.add_argument(
        "--provider",
        default="mock",
        help=(
            "provider spec: mock | anthropic:<model> | "
            "openai:<model>[@<base_url>]  (default: mock)"
        ),
    )
    p.add_argument(
        "--judge",
        default=None,
        help="provider spec for llm_judge checks (default: same as --provider)",
    )
    p.add_argument("--workers", type=int, default=4, help="parallel workers")
    p.add_argument(
        "--tags",
        default="",
        help="comma-separated tags to include (default: all)",
    )
    p.add_argument(
        "--exclude-tags",
        default="",
        help="comma-separated tags to skip",
    )
    p.add_argument(
        "--out",
        default="reports",
        help="directory for results.json, report.md and junit.xml",
    )
    p.add_argument(
        "--baseline",
        default=None,
        help="previous results.json to detect regressions against",
    )
    p.add_argument(
        "--min-pass-rate",
        type=float,
        default=0.9,
        help="gate threshold, 0.0-1.0 (default: 0.9)",
    )
    p.add_argument(
        "--allow-regressions",
        action="store_true",
        help="report regressions but do not fail the gate on them",
    )
    p.add_argument(
        "--no-gate",
        action="store_true",
        help="always exit 0; just produce reports",
    )
    p.add_argument(
        "--save-baseline",
        default=None,
        help="also write this run's results to the given path",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llmqa",
        description="Regression testing and release gating for LLM applications.",
    )
    parser.add_argument("--version", action="version", version=f"llmqa {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run suites and gate the result")
    _add_run_args(run_p)

    gate_p = sub.add_parser("gate", help="re-evaluate a stored results.json")
    gate_p.add_argument("results", help="results.json from a previous run")
    gate_p.add_argument("--baseline", default=None)
    gate_p.add_argument("--min-pass-rate", type=float, default=0.9)
    gate_p.add_argument("--allow-regressions", action="store_true")

    list_p = sub.add_parser("list", help="show the cases a suite contains")
    list_p.add_argument("suites", nargs="+")
    list_p.add_argument("--tags", default="")
    list_p.add_argument("--exclude-tags", default="")

    return parser


def _tag_set(raw: str) -> set[str]:
    return {t.strip() for t in raw.split(",") if t.strip()}


def cmd_run(args: argparse.Namespace) -> int:
    try:
        suites = load_suites(args.suites)
    except SuiteError as exc:
        print(f"suite error: {exc}", file=sys.stderr)
        return 2

    include, exclude = _tag_set(args.tags), _tag_set(args.exclude_tags)
    if include or exclude:
        suites = [s.filter_tags(include, exclude) for s in suites]
        suites = [s for s in suites if s.cases]
        if not suites:
            print("no cases matched the tag filters", file=sys.stderr)
            return 2

    try:
        provider = build_provider(args.provider)
        judge = Judge(build_provider(args.judge)) if args.judge else Judge(provider)
    except ProviderError as exc:
        print(f"provider error: {exc}", file=sys.stderr)
        return 2

    total = sum(len(s.cases) for s in suites)
    print(
        f"running {total} case(s) from {len(suites)} suite(s) "
        f"against {provider.describe()}"
    )

    result = Runner(provider, judge=judge, workers=args.workers).run(suites)

    baseline = load_baseline(args.baseline) if args.baseline else None
    if args.baseline and baseline is None:
        print(
            f"note: no usable baseline at {args.baseline}; "
            "regression checking skipped",
            file=sys.stderr,
        )

    decision = evaluate(
        result,
        threshold=args.min_pass_rate,
        baseline=baseline,
        allow_regressions=args.allow_regressions,
    )

    out = Path(args.out)
    json_path = write_json(result, out / "results.json")
    md_path = write_markdown(result, out / "report.md", decision)
    xml_path = write_junit(result, out / "junit.xml")

    if args.save_baseline:
        write_json(result, args.save_baseline)

    print(render_console(result, decision))
    print(f"  reports    {json_path}, {md_path}, {xml_path}\n")

    return 0 if args.no_gate else decision.exit_code


def cmd_gate(args: argparse.Namespace) -> int:
    path = Path(args.results)
    try:
        result = RunResult.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        print(f"cannot read results file {path}: {exc}", file=sys.stderr)
        return 2

    baseline = load_baseline(args.baseline) if args.baseline else None
    decision = evaluate(
        result,
        threshold=args.min_pass_rate,
        baseline=baseline,
        allow_regressions=args.allow_regressions,
    )
    print(render_console(result, decision))
    return decision.exit_code


def cmd_list(args: argparse.Namespace) -> int:
    try:
        suites = load_suites(args.suites)
    except SuiteError as exc:
        print(f"suite error: {exc}", file=sys.stderr)
        return 2

    include, exclude = _tag_set(args.tags), _tag_set(args.exclude_tags)
    for suite in suites:
        shown = suite.filter_tags(include, exclude) if (include or exclude) else suite
        if not shown.cases:
            continue
        print(f"\n{suite.name}  ({len(shown.cases)} cases)  {suite.path}")
        if suite.description:
            print(f"  {suite.description}")
        for case in shown.cases:
            tags = f"  [{', '.join(case.tags)}]" if case.tags else ""
            checks = ", ".join(c.type for c in case.checks)
            print(f"  {case.severity:<8} {case.id:<30} {checks}{tags}")
    print()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return cmd_run(args)
    if args.command == "gate":
        return cmd_gate(args)
    if args.command == "list":
        return cmd_list(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
