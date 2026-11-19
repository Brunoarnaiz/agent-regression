"""How agent-regression talks to your agent (and to the optional judge).

A target is either:

* a shell command (``python3 agent.py``, ``node agent.js``, ``./run.sh``...).
  It receives one JSON object on stdin, ``{"case_id", "attempt", "input"}``,
  and prints its answer on stdout, either as plain text or as JSON:
  ``{"output": "...", "tool_calls": [{"name": "...", "args": {...}}]}``

* a Python function, written as ``{python: "agent.py:respond"}`` or
  ``{python: "my_package.agent:respond"}``. It is called with the case input
  and returns a string or the same dict as above.

The judge works the same way: a command receives
``{"criterion", "input", "output"}`` and prints ``{"pass": bool, "reason": str}``;
a function is called as ``judge(criterion, input, output)``.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


class TargetError(RuntimeError):
    """The target is misconfigured or returned something unusable."""


@dataclass
class ToolCall:
    name: str
    args: dict = field(default_factory=dict)


@dataclass
class AgentResponse:
    output: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    latency_ms: float = 0.0
    error: str | None = None


class CommandTarget:
    def __init__(self, command: str, cwd: Path, timeout: float):
        self.command = command
        self.cwd = cwd
        self.timeout = timeout

    def describe(self) -> str:
        return self.command

    def run(self, payload: dict) -> str:
        # The suite is trusted config (like a Makefile), so the command goes
        # through the shell: pipes, env vars and `cd x && ...` all work.
        # It gets its own process group so a timeout kills the whole pipeline.
        proc = subprocess.Popen(
            self.command,
            shell=True,
            cwd=self.cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            start_new_session=hasattr(os, "killpg"),
        )
        try:
            stdout, stderr = proc.communicate(json.dumps(payload, ensure_ascii=False), timeout=self.timeout)
        except subprocess.TimeoutExpired:
            if hasattr(os, "killpg"):
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
            proc.communicate()
            raise TargetError(f"timed out after {self.timeout:g}s") from None
        if proc.returncode != 0:
            lines = stderr.strip().splitlines()
            tail = lines[-1] if lines else "no stderr"
            raise TargetError(f"exit code {proc.returncode}: {tail}")
        return stdout.strip()


class PythonTarget:
    def __init__(self, ref: str, base_dir: Path):
        self.ref = ref
        self.fn = _load_callable(ref, base_dir)

    def describe(self) -> str:
        return f"python:{self.ref}"

    def run(self, *args: Any) -> Any:
        return self.fn(*args)


def resolve_target(spec: str | dict, base_dir: Path, timeout: float) -> CommandTarget | PythonTarget:
    if isinstance(spec, str) and spec.strip():
        return CommandTarget(spec, base_dir, timeout)
    if isinstance(spec, dict) and len(spec) >= 1:
        if "command" in spec:
            return CommandTarget(str(spec["command"]), base_dir, float(spec.get("timeout", timeout)))
        if "python" in spec:
            return PythonTarget(str(spec["python"]), base_dir)
    raise TargetError("a target must be a command string, {command: ...} or {python: 'file.py:function'}")


class Agent:
    """Calls the agent under test and normalises whatever it returns."""

    def __init__(self, spec: str | dict, base_dir: Path, timeout: float):
        self.target = resolve_target(spec, base_dir, timeout)

    def describe(self) -> str:
        return self.target.describe()

    def __call__(self, case_id: str, input: Any, attempt: int) -> AgentResponse:
        start = time.perf_counter()
        try:
            if isinstance(self.target, CommandTarget):
                raw = self.target.run({"case_id": case_id, "attempt": attempt, "input": input})
            else:
                raw = self.target.run(input)
            output, calls = parse_response(raw)
            error = None
        except Exception as exc:
            output, calls = "", []
            error = str(exc) if isinstance(exc, TargetError) else f"{type(exc).__name__}: {exc}"
        latency = (time.perf_counter() - start) * 1000
        return AgentResponse(output, calls, latency, error)


class Judge:
    """Asks a second model (or any script) whether an output meets a criterion."""

    def __init__(self, spec: str | dict, base_dir: Path, timeout: float):
        self.target = resolve_target(spec, base_dir, timeout)

    def __call__(self, criterion: str, input: Any, output: str) -> tuple[bool, str]:
        if isinstance(self.target, CommandTarget):
            raw = self.target.run({"criterion": criterion, "input": input, "output": output})
        else:
            raw = self.target.run(criterion, input, output)
        return parse_verdict(raw)


def parse_response(raw: Any) -> tuple[str, list[ToolCall]]:
    """Accept plain text, or a dict with "output" and optional "tool_calls"."""
    if isinstance(raw, str):
        data = _maybe_json(raw)
        if isinstance(data, dict) and "output" in data:
            raw = data
        else:
            return raw, []
    if isinstance(raw, dict):
        if "output" not in raw:
            # A JSON answer without the protocol envelope: the JSON *is* the output.
            return json.dumps(raw, ensure_ascii=False), []
        output = raw["output"]
        if not isinstance(output, str):
            output = json.dumps(output, ensure_ascii=False)
        calls = [_tool_call(c) for c in raw.get("tool_calls") or []]
        return output, calls
    raise TargetError(f"agent returned {type(raw).__name__}; expected text or a dict with 'output'")


def parse_verdict(raw: Any) -> tuple[bool, str]:
    if isinstance(raw, str):
        raw = _maybe_json(raw)
    if isinstance(raw, bool):
        return raw, ""
    if isinstance(raw, dict) and isinstance(raw.get("pass"), bool):
        return raw["pass"], str(raw.get("reason") or "")
    raise TargetError('the judge must return {"pass": true|false, "reason": "..."}')


def _tool_call(raw: Any) -> ToolCall:
    if isinstance(raw, str):
        return ToolCall(raw)
    if not isinstance(raw, dict) or "name" not in raw:
        raise TargetError(f"invalid tool call {raw!r}: expected {{'name': ..., 'args': {{...}}}}")
    # Accept the field names used by the common SDKs: args / arguments / input.
    args = next((raw[k] for k in ("args", "arguments", "input") if k in raw), None) or {}
    if isinstance(args, str):
        args = _maybe_json(args)
    if not isinstance(args, dict):
        args = {"value": args}
    return ToolCall(str(raw["name"]), args)


def _maybe_json(text: str) -> Any:
    try:
        return json.loads(text)
    except ValueError:
        return text


def _load_callable(ref: str, base_dir: Path) -> Callable:
    module_ref, sep, attr = ref.rpartition(":")
    if not sep or not module_ref or not attr:
        raise TargetError(f"python target '{ref}' must look like 'agent.py:respond' or 'package.module:respond'")

    if module_ref.endswith(".py"):
        file = (base_dir / module_ref).resolve()
        if not file.is_file():
            raise TargetError(f"python target file not found: {file}")
        if str(file.parent) not in sys.path:
            sys.path.insert(0, str(file.parent))  # let the agent import its siblings
        spec = importlib.util.spec_from_file_location(f"agent_regression_target_{file.stem}", file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        if str(base_dir) not in sys.path:
            sys.path.insert(0, str(base_dir))
        try:
            module = importlib.import_module(module_ref)
        except ImportError as exc:
            raise TargetError(f"cannot import '{module_ref}': {exc}") from None

    fn = getattr(module, attr, None)
    if not callable(fn):
        raise TargetError(f"'{attr}' not found or not callable in {module_ref}")
    return fn
