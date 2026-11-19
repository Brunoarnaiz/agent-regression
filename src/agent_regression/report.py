"""Render runs and comparisons: a terminal table and a Markdown version for PRs."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table
from rich.text import Text

from .compare import ORDER, Comparison, Status
from .runner import CaseResult, RunResult

STYLE = {
    Status.REGRESSED: ("✗ REGRESSED", "bold red"),
    Status.FIXED: ("✓ FIXED", "bold green"),
    Status.FAILING: ("✗ failing", "yellow"),
    Status.NEW: ("+ new", "cyan"),
    Status.REMOVED: ("- removed", "dim"),
    Status.PASSING: ("· passing", "dim"),
}

EMOJI = {
    Status.REGRESSED: "🔴 regressed",
    Status.FIXED: "🟢 fixed",
    Status.FAILING: "🟡 failing",
    Status.NEW: "🆕 new",
    Status.REMOVED: "⚪ removed",
    Status.PASSING: "✅ passing",
}


def _rate(case: CaseResult | None) -> str:
    return f"{case.passes}/{case.total}" if case else "—"


def _ms(value: float | None) -> str:
    return "—" if value is None else f"{value:.0f} ms"


def _delta(value: float | None) -> str:
    return "" if value is None else f"{round(value):+d} ms"  # round() first: never "-0 ms"


def _run_name(run: RunResult) -> str:
    return run.label or run.target


# --- terminal ---------------------------------------------------------------


def print_run(run: RunResult, console: Console) -> None:
    table = Table(box=None, pad_edge=False, header_style="bold", title_justify="left",
                  title=f"[bold]{run.suite}[/] · {_run_name(run)}")
    table.add_column("")
    table.add_column("CASE")
    table.add_column("PASS", justify="right")
    table.add_column("LATENCY", justify="right")
    table.add_column("DETAIL", style="dim", overflow="fold")
    for case in run.cases.values():
        mark = Text("✓", style="green") if case.passed else Text("✗", style="bold red")
        table.add_row(mark, case.case_id, _rate(case), _ms(case.latency_ms),
                      "" if case.passed else case.first_failure())
    console.print(table)
    console.print()
    summary = Text()
    summary.append(f"{run.passed} passed", style="green")
    summary.append(" · ")
    summary.append(f"{run.failed} failed", style="bold red" if run.failed else "dim")
    summary.append(f"  ({run.duration_s:.1f}s)", style="dim")
    console.print(summary)


def print_comparison(cmp: Comparison, console: Console) -> None:
    title = (f"[bold]{cmp.current.suite}[/] · {_run_name(cmp.current)} "
             f"[dim]vs baseline ({_run_name(cmp.baseline)})[/]")
    table = Table(box=None, pad_edge=False, header_style="bold", title=title, title_justify="left")
    table.add_column("STATUS")
    table.add_column("CASE")
    table.add_column("BEFORE", justify="right")
    table.add_column("NOW", justify="right")
    table.add_column("Δ LATENCY", justify="right", style="dim")
    table.add_column("DETAIL", overflow="fold")
    for d in cmp.diffs:
        label, style = STYLE[d.status]
        detail_style = "red" if d.status is Status.REGRESSED else "dim"
        table.add_row(Text(label, style=style), d.case_id, _rate(d.before), _rate(d.after),
                      _delta(d.latency_delta_ms), Text(d.note, style=detail_style))
    console.print(table)
    console.print()
    console.print(summary_text(cmp))


def summary_text(cmp: Comparison) -> Text:
    text = Text()
    for status in ORDER:
        n = cmp.count(status)
        if not n:
            continue
        if text:
            text.append(" · ", style="dim")
        word = ("regression" if n == 1 else "regressions") if status is Status.REGRESSED else status.value
        text.append(f"{n} {word}", style=STYLE[status][1])
    return text


# --- markdown ---------------------------------------------------------------


def run_markdown(run: RunResult) -> str:
    lines = [f"### agent-regression · {run.suite} · {_run_name(run)}", "",
             f"**{run.passed} passed · {run.failed} failed**", "",
             "| | Case | Pass | Latency | Detail |", "|---|---|---:|---:|---|"]
    for case in run.cases.values():
        mark = "✅" if case.passed else "❌"
        detail = "" if case.passed else _md(case.first_failure())
        lines.append(f"| {mark} | `{case.case_id}` | {_rate(case)} | {_ms(case.latency_ms)} | {detail} |")
    return "\n".join(lines) + "\n"


def comparison_markdown(cmp: Comparison) -> str:
    counts = " · ".join(f"{cmp.count(s)} {s.value}" for s in ORDER if cmp.count(s))
    lines = [f"### agent-regression · {cmp.current.suite}", "",
             f"`{_run_name(cmp.current)}` vs baseline `{_run_name(cmp.baseline)}` — **{counts}**", "",
             "| Status | Case | Before | Now | Δ latency | Detail |", "|---|---|---:|---:|---:|---|"]
    for d in cmp.diffs:
        lines.append(f"| {EMOJI[d.status]} | `{d.case_id}` | {_rate(d.before)} | {_rate(d.after)} "
                     f"| {_delta(d.latency_delta_ms)} | {_md(d.note)} |")
    return "\n".join(lines) + "\n"


def _md(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")
