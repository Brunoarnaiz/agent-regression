from agent_regression.checks import CheckResult
from agent_regression.compare import Status, compare
from agent_regression.report import comparison_markdown
from agent_regression.runner import AttemptResult, CaseResult, RunResult


def case(case_id: str, passes: int, total: int = 3, min_rate: float = 1.0) -> CaseResult:
    attempts = [
        AttemptResult(attempt=i, passed=i < passes, latency_ms=100.0,
                      failures=[] if i < passes else [CheckResult("contains \"x\"", False, 'missing "x"')])
        for i in range(total)
    ]
    return CaseResult(case_id, min_rate, attempts)


def run(*cases: CaseResult, label: str = "run") -> RunResult:
    return RunResult(suite="s", target="agent", started_at="2026-11-19T12:00:00+00:00",
                     duration_s=1.0, cases={c.case_id: c for c in cases}, label=label)


def statuses(cmp) -> dict:
    return {d.case_id: d.status for d in cmp.diffs}


def test_every_status():
    baseline = run(case("regressed", 3), case("fixed", 0), case("failing", 1), case("passing", 3), case("gone", 3))
    current = run(case("regressed", 0), case("fixed", 3), case("failing", 0), case("passing", 3), case("new", 3))
    assert statuses(compare(baseline, current)) == {
        "regressed": Status.REGRESSED,
        "fixed": Status.FIXED,
        "failing": Status.FAILING,
        "passing": Status.PASSING,
        "gone": Status.REMOVED,
        "new": Status.NEW,
    }


def test_regressions_are_listed_first():
    baseline = run(case("a", 3), case("b", 3), case("c", 0))
    current = run(case("a", 3), case("b", 0), case("c", 3))
    assert [d.case_id for d in compare(baseline, current).diffs] == ["b", "c", "a"]


def test_has_regressions():
    assert compare(run(case("a", 3)), run(case("a", 2))).has_regressions
    assert not compare(run(case("a", 0)), run(case("a", 0))).has_regressions


def test_pass_rate_threshold_decides_pass():
    # 2/3 passes when only 0.66 is required, so it is not a regression...
    lenient = compare(run(case("a", 3, min_rate=0.66)), run(case("a", 2, min_rate=0.66)))
    assert statuses(lenient)["a"] is Status.PASSING
    # ...but the drop is still reported.
    assert "pass rate dropped (3/3 → 2/3)" in lenient.diffs[0].note


def test_regression_note_explains_the_failure():
    cmp = compare(run(case("a", 3)), run(case("a", 0)))
    assert cmp.diffs[0].note == 'missing "x"'


def test_markdown_report():
    cmp = compare(run(case("a", 3), label="v1"), run(case("a", 0), label="v2"))
    md = comparison_markdown(cmp)
    assert "`v2` vs baseline `v1`" in md
    assert "🔴 regressed" in md
    assert "| `a` | 3/3 | 0/3 |" in md
