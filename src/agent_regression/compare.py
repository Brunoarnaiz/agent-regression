"""Compare a run against the baseline: what broke, what got fixed."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .runner import CaseResult, RunResult


class Status(str, Enum):
    REGRESSED = "regressed"  # passed in the baseline, fails now
    FIXED = "fixed"          # failed in the baseline, passes now
    FAILING = "failing"      # failed before, still fails
    NEW = "new"              # case not in the baseline
    REMOVED = "removed"      # case no longer in the suite
    PASSING = "passing"      # passed before, still passes


# Report order: the things you need to look at first.
ORDER = [Status.REGRESSED, Status.FIXED, Status.FAILING, Status.NEW, Status.REMOVED, Status.PASSING]


@dataclass
class CaseDiff:
    case_id: str
    status: Status
    before: CaseResult | None
    after: CaseResult | None
    note: str = ""

    @property
    def latency_delta_ms(self) -> float | None:
        if self.before and self.after:
            return self.after.latency_ms - self.before.latency_ms
        return None


@dataclass
class Comparison:
    baseline: RunResult
    current: RunResult
    diffs: list[CaseDiff] = field(default_factory=list)

    def count(self, status: Status) -> int:
        return sum(d.status is status for d in self.diffs)

    @property
    def has_regressions(self) -> bool:
        return self.count(Status.REGRESSED) > 0


def compare(baseline: RunResult, current: RunResult) -> Comparison:
    diffs: list[CaseDiff] = []
    for case_id, after in current.cases.items():
        before = baseline.cases.get(case_id)
        diffs.append(_diff(case_id, before, after))
    for case_id, before in baseline.cases.items():
        if case_id not in current.cases:
            diffs.append(CaseDiff(case_id, Status.REMOVED, before, None))

    position = {s: i for i, s in enumerate(ORDER)}
    diffs.sort(key=lambda d: position[d.status])  # stable: keeps suite order within a status
    return Comparison(baseline, current, diffs)


def _diff(case_id: str, before: CaseResult | None, after: CaseResult) -> CaseDiff:
    if before is None:
        return CaseDiff(case_id, Status.NEW, None, after, after.first_failure())
    if before.passed and not after.passed:
        return CaseDiff(case_id, Status.REGRESSED, before, after, after.first_failure())
    if not before.passed and after.passed:
        return CaseDiff(case_id, Status.FIXED, before, after)
    if not after.passed:
        return CaseDiff(case_id, Status.FAILING, before, after, after.first_failure())
    note = ""
    if after.pass_rate < before.pass_rate:
        note = f"pass rate dropped ({before.passes}/{before.total} → {after.passes}/{after.total})"
    return CaseDiff(case_id, Status.PASSING, before, after, note)
