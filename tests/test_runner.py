import json
import shutil
import sys
import textwrap
from pathlib import Path

import pytest

from agent_regression.cli import main
from agent_regression.runner import Store, load_run, run_suite
from agent_regression.suite import SuiteError, load_suite

PYTHON = sys.executable
EXAMPLES = Path(__file__).resolve().parent.parent / "examples"

AGENT = '''
calls = {"n": 0}

def respond(text):
    if text == "boom":
        raise RuntimeError("model unavailable")
    if text == "flaky":
        calls["n"] += 1
        return "ok" if calls["n"] % 2 else "nope"
    if text == "order":
        return {"output": "shipped", "tool_calls": [{"name": "lookup_order", "input": {"order_id": "1"}}]}
    return text.upper()

def judge(criterion, input, output):
    return {"pass": output.isupper(), "reason": "not shouting"}
'''


def write_suite(tmp_path: Path, body: str) -> Path:
    (tmp_path / "agent.py").write_text(AGENT)
    path = tmp_path / "suite.yaml"
    path.write_text(textwrap.dedent(body))
    return path


def test_python_target_with_judge_and_tool_calls(tmp_path):
    suite = load_suite(write_suite(tmp_path, """
        target: {python: "agent.py:respond"}
        judge: {python: "agent.py:judge"}
        cases:
          - {id: shout, input: "hello", checks: [{contains: "HELLO"}, {judge: "shouts"}]}
          - {id: order, input: "order", checks: [{tool_called: lookup_order, args: {order_id: "1"}}]}
    """))
    run = run_suite(suite)
    assert run.cases["shout"].passed
    assert run.cases["order"].passed
    assert run.cases["order"].attempts[0].tool_calls == [{"name": "lookup_order", "args": {"order_id": "1"}}]


def test_agent_errors_fail_the_case_without_stopping_the_run(tmp_path):
    suite = load_suite(write_suite(tmp_path, """
        target: {python: "agent.py:respond"}
        cases:
          - {id: crash, input: "boom", checks: [{contains: "x"}]}
          - {id: fine, input: "hi", checks: [{contains: "HI"}]}
    """))
    run = run_suite(suite)
    assert not run.cases["crash"].passed
    assert run.cases["crash"].first_failure() == "agent error: RuntimeError: model unavailable"
    assert run.cases["fine"].passed


def test_repeat_and_pass_rate(tmp_path):
    suite = load_suite(write_suite(tmp_path, """
        target: {python: "agent.py:respond"}
        repeat: 4
        cases:
          - {id: strict, input: "flaky", checks: [{equals: "ok"}]}
          - {id: lenient, input: "flaky", min_pass_rate: 0.5, checks: [{equals: "ok"}]}
    """))
    run = run_suite(suite, jobs=1)
    assert run.cases["strict"].total == 4
    assert (run.cases["strict"].passes, run.cases["strict"].passed) == (2, False)
    assert (run.cases["lenient"].passes, run.cases["lenient"].passed) == (2, True)


def test_command_target_protocol(tmp_path):
    (tmp_path / "echo_agent.py").write_text(
        "import json, sys\n"
        "req = json.load(sys.stdin)\n"
        "print(json.dumps({'output': f\"{req['case_id']}:{req['attempt']}:{req['input']}\"}))\n"
    )
    suite = load_suite(write_suite(tmp_path, f"""
        target: "{PYTHON} echo_agent.py"
        cases:
          - {{id: echo, input: "hi", checks: [{{equals: "echo:0:hi"}}]}}
    """))
    assert run_suite(suite).cases["echo"].passed


def test_command_target_plain_text_and_failures(tmp_path):
    suite = load_suite(write_suite(tmp_path, f"""
        target: "echo plain text"
        timeout: 0.5
        cases:
          - {{id: text, input: "x", checks: [{{equals: "plain text"}}]}}
    """))
    assert run_suite(suite).cases["text"].passed

    failing = run_suite(suite, target="echo oops >&2; exit 3")
    assert failing.cases["text"].first_failure() == "agent error: exit code 3: oops"

    slow = run_suite(suite, target=f'{PYTHON} -c "import time; time.sleep(5)"')
    assert slow.cases["text"].first_failure() == "agent error: timed out after 0.5s"


def test_run_files_round_trip(tmp_path):
    suite = load_suite(write_suite(tmp_path, """
        target: {python: "agent.py:respond"}
        cases: [{id: a, input: "hi", checks: [{contains: "HI"}]}]
    """))
    store = Store(suite)
    run = run_suite(suite, label="v1")
    path = store.save_run(run)
    assert path.name.endswith("-v1.json")
    assert load_run(path).to_dict() == run.to_dict()
    store.set_baseline(path)
    assert store.load_baseline().label == "v1"


@pytest.mark.parametrize(
    "body, message",
    [
        ("cases: [{id: a, input: x, checks: [{contains: x}]}]", "missing 'target'"),
        ("target: x\nrepet: 3\ncases: [{id: a, input: x, checks: [{contains: x}]}]", "did you mean 'repeat'"),
        ("target: x\ncases: [{id: a, input: x, checks: [{contains: x}]}, {id: a, input: y, checks: [{contains: y}]}]",
         "duplicate id"),
        ("target: x\ncases: [{id: a, checks: [{contains: x}]}]", "missing 'input'"),
        ("target: x\ncases: [{id: a, input: x, checks: [{judge: polite}]}]", "no 'judge' target"),
        ("target: x\ncases: [{id: a, input: x, checks: [{contians: x}]}]", "case 'a': checks\\[0\\]"),
        ("target: x\nmin_pass_rate: 2\ncases: [{id: a, input: x, checks: [{contains: x}]}]", "between 0 and 1"),
    ],
)
def test_suite_errors_are_helpful(tmp_path, body, message):
    path = tmp_path / "suite.yaml"
    path.write_text(body)
    with pytest.raises(SuiteError, match=message):
        load_suite(path)


def test_cli_end_to_end_on_the_demo(tmp_path, capsys):
    """The support-bot demo: v1 is the baseline, v2 must be reported as a regression."""
    demo = tmp_path / "support-bot"
    shutil.copytree(EXAMPLES / "support-bot", demo, ignore=shutil.ignore_patterns(".agent-regression"))
    suite = str(demo / "suite.yaml")

    assert main(["run", suite, "--save-baseline", "-t", f"{PYTHON} agent_v1.py"]) == 0
    assert main(["run", suite, "-t", f"{PYTHON} agent_v2.py", "-l", "v2",
                 "--markdown", str(tmp_path / "report.md")]) == 1

    out = capsys.readouterr().out
    assert "REGRESSED" in out and "order-lookup" in out
    assert "FIXED" in out and "answers-in-spanish" in out
    report = (tmp_path / "report.md").read_text()
    assert "🔴 regressed | `order-lookup`" in report

    runs = list((demo / ".agent-regression" / "support-bot" / "runs").glob("*.json"))
    assert len(runs) == 2
    # Both runs can start in the same second, so find them by label, not by name order.
    v2 = next(p for p in runs if p.name.endswith("-v2.json"))
    v1 = next(p for p in runs if p != v2)
    assert main(["compare", str(v1), str(v2)]) == 1
    assert main(["baseline", suite, "--run", str(v2)]) == 0
    baseline = json.loads((demo / ".agent-regression" / "support-bot" / "baseline.json").read_text())
    assert baseline["label"] == "v2"


def test_cli_reports_config_errors_with_exit_code_2(tmp_path, capsys):
    assert main(["run", str(tmp_path / "missing.yaml")]) == 2
    assert "suite file not found" in capsys.readouterr().err
