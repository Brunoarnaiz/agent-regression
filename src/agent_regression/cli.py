"""Command-line interface.

    agent-regression run suite.yaml [--target CMD] [--save-baseline] ...
    agent-regression baseline suite.yaml [--run RUN.json]
    agent-regression compare BEFORE.json AFTER.json

Exit codes: 0 = ok, 1 = regressions (or failures with --strict), 2 = bad config.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn

from . import __version__
from .compare import compare
from .report import comparison_markdown, print_comparison, print_run, run_markdown
from .runner import RunFileError, Store, load_run, run_suite
from .suite import SuiteError, load_suite
from .targets import TargetError

EXIT_OK, EXIT_FAILED, EXIT_ERROR = 0, 1, 2


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    console = Console(highlight=False)
    try:
        return args.func(args, console)
    except (SuiteError, TargetError, RunFileError, OSError) as exc:
        Console(stderr=True, highlight=False).print(f"[bold red]error:[/] {exc}")
        return EXIT_ERROR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-regression",
        description="Regression tests for AI agents: find out what your last change broke.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a suite and compare it with the baseline")
    run.add_argument("suite", type=Path, help="path to suite.yaml")
    run.add_argument("-t", "--target", help="override the suite's target command, e.g. 'python3 agent_v2.py'")
    run.add_argument("-n", "--repeat", type=_positive_int, help="override how many times each case runs")
    run.add_argument("-j", "--jobs", type=_positive_int, default=4, help="attempts run in parallel (default: 4)")
    run.add_argument("-l", "--label", help="a name for this run, e.g. 'v2' or a git sha")
    run.add_argument("--save-baseline", action="store_true", help="make this run the new baseline")
    run.add_argument("--no-compare", action="store_true", help="don't compare with the baseline")
    run.add_argument("--strict", action="store_true", help="exit 1 if any case fails, not only on regressions")
    run.add_argument("--markdown", type=Path, metavar="FILE", help="also write the report as Markdown")
    run.set_defaults(func=cmd_run)

    base = sub.add_parser("baseline", help="make a saved run the baseline (default: the latest run)")
    base.add_argument("suite", type=Path, help="path to suite.yaml")
    base.add_argument("--run", type=Path, metavar="RUN.json", help="the run to promote")
    base.set_defaults(func=cmd_baseline)

    cmp = sub.add_parser("compare", help="compare two saved runs")
    cmp.add_argument("before", type=Path, metavar="BEFORE.json")
    cmp.add_argument("after", type=Path, metavar="AFTER.json")
    cmp.add_argument("--markdown", type=Path, metavar="FILE", help="also write the report as Markdown")
    cmp.set_defaults(func=cmd_compare)
    return parser


def cmd_run(args: argparse.Namespace, console: Console) -> int:
    suite = load_suite(args.suite)
    store = Store(suite)
    baseline = None if args.no_compare else store.load_baseline()

    total = sum(args.repeat or case.repeat for case in suite.cases)
    with Progress(TextColumn("[dim]running {task.description}"), BarColumn(), MofNCompleteColumn(),
                  console=console, transient=True) as progress:
        task = progress.add_task(suite.name, total=total)
        run = run_suite(suite, target=args.target, repeat=args.repeat, jobs=args.jobs,
                        label=args.label, on_attempt=lambda *_: progress.advance(task))
    path = store.save_run(run)

    if baseline:
        comparison = compare(baseline, run)
        print_comparison(comparison, console)
        markdown = comparison_markdown(comparison)
        failed = comparison.has_regressions
    else:
        print_run(run, console)
        markdown = run_markdown(run)
        failed = False
    if args.strict and run.failed:
        failed = True

    console.print(f"[dim]run saved to {_rel(path)}[/]")
    if args.save_baseline:
        store.set_baseline(path)
        console.print(f"[green]baseline updated[/] [dim]→ {_rel(store.baseline_path)}[/]")
        failed = args.strict and run.failed > 0  # you just accepted this run as the reference
    elif baseline is None and not args.no_compare:
        console.print("[dim]no baseline yet: run again with --save-baseline to use this as the reference[/]")
    _write_markdown(args.markdown, markdown, console)
    return EXIT_FAILED if failed else EXIT_OK


def cmd_baseline(args: argparse.Namespace, console: Console) -> int:
    suite = load_suite(args.suite)
    store = Store(suite)
    run_path = args.run or store.latest_run()
    if run_path is None:
        raise SuiteError(f"no runs yet for '{suite.name}': use `agent-regression run {args.suite}` first")
    load_run(run_path)  # make sure it is a valid run before promoting it
    store.set_baseline(Path(run_path))
    console.print(f"[green]baseline for {suite.name}[/] ← {_rel(Path(run_path))}")
    return EXIT_OK


def cmd_compare(args: argparse.Namespace, console: Console) -> int:
    comparison = compare(load_run(args.before), load_run(args.after))
    print_comparison(comparison, console)
    _write_markdown(args.markdown, comparison_markdown(comparison), console)
    return EXIT_FAILED if comparison.has_regressions else EXIT_OK


def _write_markdown(path: Path | None, markdown: str, console: Console) -> None:
    if path:
        path.write_text(markdown, encoding="utf-8")
        console.print(f"[dim]markdown report written to {_rel(path)}[/]")


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return value
