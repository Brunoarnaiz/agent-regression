"""Checks: the conditions a test case sets on an agent's response.

The life of a check:

    YAML item                   parse_check()            evaluate()
    {"contains": "30 days"}  ───────────────▶  Check  ───────────────▶  CheckResult
                             (once, when the          (on every attempt)
                              suite is loaded)

1. When the suite is loaded, `parse_check()` turns every YAML item into a
   `Check` and validates it, so a typo fails *before* the agent is called.
2. On every attempt, `evaluate()` runs the check against the agent's response
   and returns a `CheckResult`: passed or not, plus a short reason.

Every check is one small function registered under the name used in YAML.
Adding a new check = writing one function decorated with `@_register(...)`.
Nothing else in the program has to change.
"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    # Only imported for type hints. A real import here would be circular:
    # targets.py does not need checks.py, but keeping the dependency one-way
    # makes the module graph easier to follow.
    from .targets import AgentResponse

# The signature a judge must have: (criterion, case input, agent output) -> (passed, reason).
JudgeFn = Callable[[str, Any, str], "tuple[bool, str]"]


class CheckError(ValueError):
    """Raised by parse_check() when a check in the suite is malformed."""


# =============================================================================
# 1. The data: what a check is, what it returns, what it may need
# =============================================================================


@dataclass
class Check:
    """One parsed check, e.g. `- contains: "30 days"` → Check("contains", "30 days").

    kind    -- the check name ("contains", "tool_called", ...)
    value   -- what comes after the colon ("30 days", "lookup_order", ...)
    options -- extra modifiers on the same item ({"ignore_case": True}, {"args": {...}})
    """

    kind: str
    value: Any
    options: dict[str, Any] = field(default_factory=dict)  # a fresh dict per Check, never shared

    def label(self) -> str:
        """Short human-readable description for reports: `contains "30 days"`."""
        if self.kind in ("tool_called", "tool_not_called"):
            args = self.options.get("args")
            return f"{self.kind} {self.value}" + (f"({_fmt_args(args)})" if args else "")
        if self.kind == "json":
            return "json"
        value = self.value
        if isinstance(value, str):
            text = value if len(value) <= 40 else value[:37] + "..."  # keep report rows short
            shown = f"/{text}/" if "regex" in self.kind else json.dumps(text, ensure_ascii=False)
        else:
            shown = json.dumps(value, ensure_ascii=False)
        return f"{self.kind} {shown}"


@dataclass
class CheckResult:
    """The outcome of running one check on one response."""

    check: str        # the label, e.g. 'contains "30 days"'
    passed: bool
    detail: str = ""  # why it failed, e.g. 'missing "30 days"' (empty when it passed)


@dataclass
class Context:
    """What a check may need besides the response itself.

    Only the `judge` check uses it today: it needs the original question and the
    judge function. Passing a Context object (instead of more arguments) means a
    future check can get new information without changing every check function.
    """

    input: Any = None
    judge: JudgeFn | None = None


@dataclass
class _Spec:
    """How one check type is defined in the registry (internal)."""

    fn: Callable[[Any, "AgentResponse", dict, Context], "tuple[bool, str]"]  # does the checking
    options: frozenset[str]                                                 # modifiers it accepts
    validate: Callable[[Any], Any]                                          # checks the YAML value


# =============================================================================
# 2. The registry: check name in YAML → the function that implements it
# =============================================================================

REGISTRY: dict[str, _Spec] = {}


def _register(name: str, validate: Callable[[Any], Any], options: tuple[str, ...] = ()):
    """Decorator that adds a check function to REGISTRY under `name`.

        @_register("contains", _text_or_list, options=("ignore_case",))
        def _contains(value, resp, opts, ctx): ...

    is the same as writing, after the function:

        REGISTRY["contains"] = _Spec(_contains, frozenset({"ignore_case"}), _text_or_list)
    """

    def deco(fn):
        REGISTRY[name] = _Spec(fn, frozenset(options), validate)
        return fn  # return the function unchanged: registering is the only side effect

    return deco


# =============================================================================
# 3. Value validators: run at load time, they reject bad YAML values early
#    Each one returns the value if it is fine, or raises CheckError.
# =============================================================================


def _text_or_list(value: Any) -> Any:
    """"text" or ["a", "b"]: used by contains / not_contains."""
    if isinstance(value, str) and value:
        return value
    if isinstance(value, list) and value and all(isinstance(v, str) and v for v in value):
        return value
    raise CheckError("expected a non-empty string or a list of strings")


def _text_list(value: Any) -> list[str]:
    """Like _text_or_list, but always gives back a list ("a" → ["a"])."""
    if isinstance(value, str):
        value = [value]
    return _text_or_list(value)


def _text(value: Any) -> str:
    if isinstance(value, str) and value:
        return value
    raise CheckError("expected a non-empty string")


def _pattern(value: Any) -> str:
    """A string that must also compile as a regular expression."""
    _text(value)
    try:
        re.compile(value)
    except re.error as exc:
        raise CheckError(f"invalid regex: {exc}") from None
    return value


def _positive_number(value: Any) -> float:
    # bool is a subclass of int in Python (True == 1), so it has to be excluded explicitly.
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
        return value
    raise CheckError("expected a positive number")


def _true(value: Any) -> bool:
    if value is True:
        return True
    raise CheckError("write it as `json: true`")


# =============================================================================
# 4. Small helpers shared by the checks
# =============================================================================


def _as_list(value: Any) -> list[str]:
    """Lets a check treat "x" and ["x", "y"] the same way."""
    return [value] if isinstance(value, str) else list(value)


def _fold(text: str, opts: dict) -> str:
    """Lower-case the text if the check has `ignore_case: true`."""
    return text.lower() if opts.get("ignore_case") else text


def _quote(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)  # "30 days" with quotes, accents kept as-is


def _fmt_args(args: dict) -> str:
    """{"order_id": "1234"} → order_id='1234'"""
    return ", ".join(f"{k}={v!r}" if isinstance(v, str) else f"{k}={v}" for k, v in args.items())


def _same(expected: Any, actual: Any) -> bool:
    # YAML `order_id: "1234"` should match an agent that sends the number 1234.
    return expected == actual or str(expected) == str(actual)


def _parse_json(text: str) -> tuple[Any, str]:
    """Returns (data, "") on success or (None, error message) on failure."""
    try:
        return json.loads(text), ""
    except ValueError as exc:
        return None, f"output is not valid JSON ({exc})"


# =============================================================================
# 5. The checks themselves
#
# Every check function has the same shape:
#
#     def _name(value, resp, opts, ctx) -> (passed: bool, detail: str)
#
#     value -- what was written in YAML after the check name
#     resp  -- the AgentResponse: .output (text), .tool_calls, .latency_ms
#     opts  -- the modifiers on the item, e.g. {"ignore_case": True}
#     ctx   -- the Context (only the judge uses it)
#
# `detail` explains a failure in a few words; it is "" when the check passed.
# =============================================================================

# --- text checks -------------------------------------------------------------


@_register("contains", _text_or_list, options=("ignore_case",))
def _contains(value, resp, opts, ctx):
    """Every given text must appear in the output."""
    out = _fold(resp.output, opts)
    missing = [t for t in _as_list(value) if _fold(t, opts) not in out]
    return not missing, f"missing {_quote(missing[0])}" if missing else ""


@_register("contains_any", _text_list, options=("ignore_case",))
def _contains_any(value, resp, opts, ctx):
    """At least one of the given texts must appear."""
    out = _fold(resp.output, opts)
    ok = any(_fold(t, opts) in out for t in value)
    return ok, "" if ok else f"none of {json.dumps(value, ensure_ascii=False)} found"


@_register("not_contains", _text_or_list, options=("ignore_case",))
def _not_contains(value, resp, opts, ctx):
    """None of the given texts may appear."""
    out = _fold(resp.output, opts)
    found = [t for t in _as_list(value) if _fold(t, opts) in out]
    return not found, f"found {_quote(found[0])}" if found else ""


@_register("equals", _text, options=("ignore_case",))
def _equals(value, resp, opts, ctx):
    """The whole output must be exactly this (surrounding whitespace ignored)."""
    ok = _fold(resp.output.strip(), opts) == _fold(value.strip(), opts)
    return ok, "" if ok else f"output is {_quote(resp.output.strip()[:60])}"


@_register("regex", _pattern, options=("ignore_case",))
def _regex(value, resp, opts, ctx):
    """The pattern must match somewhere in the output (re.search, not a full match)."""
    flags = re.IGNORECASE if opts.get("ignore_case") else 0
    ok = re.search(value, resp.output, flags) is not None
    return ok, "" if ok else f"no match for /{value}/"


@_register("not_regex", _pattern, options=("ignore_case",))
def _not_regex(value, resp, opts, ctx):
    """The pattern must not match anywhere, e.g. internal codes like STAFF-\\w+."""
    flags = re.IGNORECASE if opts.get("ignore_case") else 0
    match = re.search(value, resp.output, flags)
    return match is None, f"matched {_quote(match.group(0))}" if match else ""


@_register("max_chars", _positive_number)
def _max_chars(value, resp, opts, ctx):
    n = len(resp.output)
    return n <= value, f"{n} chars > {value}" if n > value else ""


# --- format checks -----------------------------------------------------------


@_register("json", _true)
def _json(value, resp, opts, ctx):
    """The output must be valid JSON."""
    _, err = _parse_json(resp.output)
    return not err, err


@_register("json_keys", _text_list)
def _json_keys(value, resp, opts, ctx):
    """The output must be a JSON object containing these keys."""
    data, err = _parse_json(resp.output)
    if err:
        return False, err
    if not isinstance(data, dict):
        return False, "output JSON is not an object"
    missing = [k for k in value if k not in data]
    return not missing, f"missing key {_quote(missing[0])}" if missing else ""


# --- agent checks: what the agent *did*, not what it said ---------------------


@_register("tool_called", _text, options=("args",))
def _tool_called(value, resp, opts, ctx):
    """The agent must have called this tool (optionally with at least these args)."""
    calls = [c for c in resp.tool_calls if c.name == value]
    if not calls:
        called = ", ".join(c.name for c in resp.tool_calls) or "none"
        return False, f"{value} not called (called: {called})"
    expected = opts.get("args")
    if not expected:
        return True, ""
    # The tool may have been called several times; one call with the right args is enough.
    # Extra args the agent sent are fine: we only require the ones written in YAML.
    for call in calls:
        if all(k in call.args and _same(v, call.args[k]) for k, v in expected.items()):
            return True, ""
    return False, f"{value} called with ({_fmt_args(calls[0].args)})"


@_register("tool_not_called", _text)
def _tool_not_called(value, resp, opts, ctx):
    """The agent must NOT have called this tool (e.g. apply_discount)."""
    called = any(c.name == value for c in resp.tool_calls)
    return not called, f"{value} was called" if called else ""


@_register("max_latency_ms", _positive_number)
def _max_latency(value, resp, opts, ctx):
    ms = resp.latency_ms
    return ms <= value, f"{ms:.0f} ms > {value} ms" if ms > value else ""


@_register("judge", _text)
def _judge(value, resp, opts, ctx):
    """Ask the judge (usually another LLM) whether the output meets a criterion."""
    if ctx.judge is None:
        return False, "no judge configured in the suite"
    passed, reason = ctx.judge(value, ctx.input, resp.output)
    return passed, "" if passed else (reason or "judge said no")


# =============================================================================
# 6. Parsing (YAML → Check) and evaluation (Check + response → CheckResult)
# =============================================================================

# The type each modifier must have in YAML.
_OPTION_TYPES = {"ignore_case": bool, "args": dict}


def did_you_mean(word: str, options) -> str:
    """' (did you mean 'contains'?)' for a near miss like 'contians', or ''."""
    match = difflib.get_close_matches(word, list(options), n=1, cutoff=0.6)
    return f" (did you mean '{match[0]}'?)" if match else ""


def parse_check(raw: Any) -> Check:
    """Turn one YAML check item into a validated Check.

    Rules for an item:
      * it is a mapping (`- contains: "x"`, not just `- "x"`)
      * it has exactly one check name
      * every other key is an option that check accepts, with the right type
      * the value passes the check's validator
    """
    if not isinstance(raw, dict) or not raw:
        raise CheckError("a check must be a mapping, e.g. `- contains: \"text\"`")

    # Which keys of the item are check names?
    kinds = [k for k in raw if k in REGISTRY]
    if not kinds:
        key = next(iter(raw))  # the first key: most likely the misspelled check name
        raise CheckError(f"unknown check '{key}'{did_you_mean(key, REGISTRY)}")
    if len(kinds) > 1:
        raise CheckError(f"one check per item, found {' and '.join(kinds)}; split them into separate list items")

    kind = kinds[0]
    spec = REGISTRY[kind]

    # Everything that isn't the check name must be an option this check accepts.
    options = {k: v for k, v in raw.items() if k != kind}
    for key, val in options.items():
        if key not in spec.options:
            valid = ", ".join(sorted(spec.options)) or "none"
            raise CheckError(f"'{key}' is not an option of '{kind}' (valid options: {valid})")
        if not isinstance(val, _OPTION_TYPES[key]):
            raise CheckError(f"'{key}' must be a {_OPTION_TYPES[key].__name__}")

    try:
        value = spec.validate(raw[kind])
    except CheckError as exc:
        raise CheckError(f"'{kind}': {exc}") from None  # prefix the check name for a clearer message
    return Check(kind, value, options)


def evaluate(check: Check, response: "AgentResponse", ctx: Context) -> CheckResult:
    """Run one check against one response.

    A check that raises (typically a judge whose API call failed) counts as a
    failed check with the error as its detail, instead of crashing the whole run.
    """
    try:
        passed, detail = REGISTRY[check.kind].fn(check.value, response, check.options, ctx)
    except Exception as exc:
        passed, detail = False, f"check crashed: {type(exc).__name__}: {exc}"
    return CheckResult(check.label(), bool(passed), detail)
