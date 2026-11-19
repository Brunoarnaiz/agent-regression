"""Run a suite against an agent, and keep the history of runs on disk.

    .agent-regression/<suite-name>/
        runs/20261119-153012-v2.json    every run, newest last
        baseline.json                   the run you compare against
"""

from __future__ import annotations

import json
import re
import shutil
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .checks import CheckResult, Context, evaluate
from .suite import Case, Suite
from .targets import Agent, Judge

SCHEMA_VERSION = 1
STORE_DIR = ".agent-regression"
MAX_OUTPUT_CHARS = 2000


@dataclass
class AttemptResult:
    attempt: int
    passed: bool
    latency_ms: float
    output: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    failures: list[CheckResult] = field(default_factory=list)
    error: str | None = None


@dataclass
class CaseResult:
    case_id: str
    min_pass_rate: float
    attempts: list[AttemptResult]

    @property
    def passes(self) -> int:
        return sum(a.passed for a in self.attempts)

    @property
    def total(self) -> int:
        return len(self.attempts)

    @property
    def pass_rate(self) -> float:
        return self.passes / self.total if self.total else 0.0

    @property
    def passed(self) -> bool:
        return self.total > 0 and self.pass_rate + 1e-9 >= self.min_pass_rate

    @property
    def latency_ms(self) -> float:
        return statistics.median(a.latency_ms for a in self.attempts) if self.attempts else 0.0

    def first_failure(self) -> str:
        """The most useful one-line reason this case failed."""
        for attempt in self.attempts:
            for failure in attempt.failures:
                return failure.detail or f"{failure.check} failed"
        return ""


@dataclass
class RunResult:
    suite: str
    target: str
    started_at: str
    duration_s: float
    cases: dict[str, CaseResult]
    label: str | None = None

    @property
    def passed(self) -> int:
        return sum(c.passed for c in self.cases.values())

    @property
    def failed(self) -> int:
        return len(self.cases) - self.passed

    def to_dict(self) -> dict:
        return {
            "schema": SCHEMA_VERSION,
            "suite": self.suite,
            "label": self.label,
            "target": self.target,
            "started_at": self.started_at,
            "duration_s": self.duration_s,
            "cases": [asdict(c) for c in self.cases.values()],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RunResult":
        cases = {}
        for c in data["cases"]:
            attempts = [
                AttemptResult(**{**a, "failures": [CheckResult(**f) for f in a.get("failures", [])]})
                for a in c["attempts"]
            ]
            cases[c["case_id"]] = CaseResult(c["case_id"], c["min_pass_rate"], attempts)
        return cls(
            suite=data["suite"],
            target=data["target"],
            started_at=data["started_at"],
            duration_s=data["duration_s"],
            cases=cases,
            label=data.get("label"),
        )


def run_suite(
    suite: Suite,
    *,
    target: str | dict | None = None,
    repeat: int | None = None,
    jobs: int = 4,
    label: str | None = None,
    on_attempt: Callable[[str, AttemptResult], None] | None = None,
) -> RunResult:
    """Run every case `repeat` times (cases and attempts run in parallel)."""
    agent = Agent(target or suite.target, suite.base_dir, suite.timeout)
    judge = Judge(suite.judge, suite.base_dir, suite.timeout) if suite.judge else None

    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    attempts: dict[str, list[AttemptResult]] = {case.id: [] for case in suite.cases}

    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        futures = {
            pool.submit(_run_attempt, agent, judge, case, i): case.id
            for case in suite.cases
            for i in range(repeat or case.repeat)
        }
        for future in as_completed(futures):
            case_id = futures[future]
            result = future.result()
            attempts[case_id].append(result)
            if on_attempt:
                on_attempt(case_id, result)

    cases = {
        case.id: CaseResult(case.id, case.min_pass_rate, sorted(attempts[case.id], key=lambda a: a.attempt))
        for case in suite.cases
    }
    return RunResult(
        suite=suite.name,
        target=agent.describe(),
        started_at=started.isoformat(timespec="seconds"),
        duration_s=round(time.perf_counter() - t0, 2),
        cases=cases,
        label=label,
    )


def _run_attempt(agent: Agent, judge: Judge | None, case: Case, attempt: int) -> AttemptResult:
    response = agent(case.id, case.input, attempt)
    if response.error:
        failures = [CheckResult("agent", False, f"agent error: {response.error}")]
    else:
        ctx = Context(input=case.input, judge=judge)
        failures = [r for r in (evaluate(c, response, ctx) for c in case.checks) if not r.passed]
    return AttemptResult(
        attempt=attempt,
        passed=not failures,
        latency_ms=round(response.latency_ms, 1),
        output=response.output[:MAX_OUTPUT_CHARS],
        tool_calls=[asdict(tc) for tc in response.tool_calls],
        failures=failures,
        error=response.error,
    )


# --- history on disk --------------------------------------------------------


class Store:
    """Where runs and the baseline of one suite live."""

    def __init__(self, suite: Suite):
        self.root = suite.base_dir / STORE_DIR / _slug(suite.name)
        self.runs_dir = self.root / "runs"
        self.baseline_path = self.root / "baseline.json"

    def save_run(self, run: RunResult) -> Path:
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.fromisoformat(run.started_at).astimezone().strftime("%Y%m%d-%H%M%S")  # local time
        stem = stamp + (f"-{_slug(run.label)}" if run.label else "")
        path = self.runs_dir / f"{stem}.json"
        n = 2
        while path.exists():
            path = self.runs_dir / f"{stem}-{n}.json"
            n += 1
        save_run(run, path)
        return path

    def latest_run(self) -> Path | None:
        runs = sorted(self.runs_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
        return runs[-1] if runs else None

    def load_baseline(self) -> RunResult | None:
        return load_run(self.baseline_path) if self.baseline_path.exists() else None

    def set_baseline(self, run_path: Path) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(run_path, self.baseline_path)
        return self.baseline_path


def save_run(run: RunResult, path: Path) -> None:
    path.write_text(json.dumps(run.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


class RunFileError(ValueError):
    """A file that should hold a saved run doesn't."""


def load_run(path: str | Path) -> RunResult:
    path = Path(path)
    text = path.read_text(encoding="utf-8")  # a missing file raises FileNotFoundError
    try:
        return RunResult.from_dict(json.loads(text))
    except (ValueError, KeyError, TypeError) as exc:
        raise RunFileError(f"{path}: not a valid agent-regression run ({exc})") from None


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-") or "suite"
