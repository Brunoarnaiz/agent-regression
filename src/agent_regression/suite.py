"""Load and validate a suite file (suite.yaml).

    name: support-bot
    target: python3 agent.py      # how to call your agent
    repeat: 3                     # run every case 3 times
    min_pass_rate: 1.0            # a case passes if >= this fraction of runs pass
    cases:
      - id: refund-policy
        input: "Can I return a product after 40 days?"
        checks:
          - contains: "30 days"
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .checks import Check, CheckError, did_you_mean, parse_check

SUITE_KEYS = {"name", "target", "judge", "repeat", "min_pass_rate", "timeout", "cases"}
CASE_KEYS = {"id", "input", "checks", "repeat", "min_pass_rate"}


class SuiteError(ValueError):
    """The suite file is missing, malformed or uses an unknown option."""


@dataclass
class Case:
    id: str
    input: Any
    checks: list[Check]
    repeat: int = 1
    min_pass_rate: float = 1.0


@dataclass
class Suite:
    name: str
    path: Path
    target: str | dict
    cases: list[Case]
    judge: str | dict | None = None
    timeout: float = 60.0

    @property
    def base_dir(self) -> Path:
        """Commands and relative paths in the suite run from here."""
        return self.path.parent


def load_suite(path: str | Path) -> Suite:
    path = Path(path)
    if not path.is_file():
        raise SuiteError(f"suite file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise SuiteError(f"{path}: invalid YAML: {exc}") from None
    if not isinstance(data, dict):
        raise SuiteError(f"{path}: expected a mapping with 'target' and 'cases'")

    where = str(path)
    _reject_unknown(data, SUITE_KEYS, where)

    target = data.get("target")
    if not isinstance(target, (str, dict)) or not target:
        raise SuiteError(f"{where}: missing 'target' (the command or function that runs your agent)")
    judge = data.get("judge")
    if judge is not None and not isinstance(judge, (str, dict)):
        raise SuiteError(f"{where}: 'judge' must be a command or a mapping like the target")

    repeat = _positive_int(data.get("repeat", 1), "repeat", where)
    min_rate = _rate(data.get("min_pass_rate", 1.0), where)
    timeout = data.get("timeout", 60)
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0:
        raise SuiteError(f"{where}: 'timeout' must be a positive number of seconds")

    raw_cases = data.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise SuiteError(f"{where}: 'cases' must be a non-empty list")

    cases: list[Case] = []
    seen: set[str] = set()
    for i, raw in enumerate(raw_cases):
        cases.append(_parse_case(raw, i, where, seen, repeat, min_rate))

    if judge is None and any(c.kind == "judge" for case in cases for c in case.checks):
        raise SuiteError(f"{where}: some cases use a 'judge' check but the suite has no 'judge' target")

    return Suite(
        name=str(data.get("name") or path.stem),
        path=path.resolve(),
        target=target,
        cases=cases,
        judge=judge,
        timeout=float(timeout),
    )


def _parse_case(raw: Any, index: int, where: str, seen: set[str], repeat: int, min_rate: float) -> Case:
    if not isinstance(raw, dict):
        raise SuiteError(f"{where}: cases[{index}] must be a mapping with 'id', 'input' and 'checks'")
    if "id" not in raw:
        raise SuiteError(f"{where}: cases[{index}] is missing 'id'")
    case_id = str(raw["id"])
    where = f"{where}: case '{case_id}'"
    _reject_unknown(raw, CASE_KEYS, where)
    if case_id in seen:
        raise SuiteError(f"{where}: duplicate id")
    seen.add(case_id)
    if "input" not in raw:
        raise SuiteError(f"{where}: missing 'input'")

    raw_checks = raw.get("checks")
    if not isinstance(raw_checks, list) or not raw_checks:
        raise SuiteError(f"{where}: 'checks' must be a non-empty list")
    checks = []
    for j, item in enumerate(raw_checks):
        try:
            checks.append(parse_check(item))
        except CheckError as exc:
            raise SuiteError(f"{where}: checks[{j}]: {exc}") from None

    return Case(
        id=case_id,
        input=raw["input"],
        checks=checks,
        repeat=_positive_int(raw.get("repeat", repeat), "repeat", where),
        min_pass_rate=_rate(raw.get("min_pass_rate", min_rate), where),
    )


def _reject_unknown(data: dict, allowed: set[str], where: str) -> None:
    for key in data:
        if key not in allowed:
            raise SuiteError(f"{where}: unknown key '{key}'{did_you_mean(str(key), allowed)}")


def _positive_int(value: Any, name: str, where: str) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 1:
        return value
    raise SuiteError(f"{where}: '{name}' must be an integer >= 1")


def _rate(value: Any, where: str) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and 0 < value <= 1:
        return float(value)
    raise SuiteError(f"{where}: 'min_pass_rate' must be a number between 0 and 1 (e.g. 0.66)")
