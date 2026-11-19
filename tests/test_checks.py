import pytest

from agent_regression.checks import CheckError, Context, evaluate, parse_check
from agent_regression.targets import AgentResponse, ToolCall


def run(raw: dict, output: str = "", tool_calls=(), latency_ms: float = 10, judge=None):
    response = AgentResponse(output=output, tool_calls=list(tool_calls), latency_ms=latency_ms)
    return evaluate(parse_check(raw), response, Context(input="question", judge=judge))


# --- text checks -------------------------------------------------------------


def test_contains_single_and_list():
    assert run({"contains": "30 days"}, "within 30 days of delivery").passed
    result = run({"contains": ["30 days", "refund"]}, "within 30 days")
    assert not result.passed
    assert result.detail == 'missing "refund"'


def test_contains_is_case_sensitive_unless_asked():
    assert not run({"contains": "hello"}, "Hello there").passed
    assert run({"contains": "hello", "ignore_case": True}, "Hello there").passed


def test_contains_any():
    assert run({"contains_any": ["envío", "envíos"]}, "Hacemos envíos").passed
    assert not run({"contains_any": ["envío"]}, "We ship").passed


def test_not_contains():
    result = run({"not_contains": "of course", "ignore_case": True}, "Of course you can!")
    assert not result.passed
    assert result.detail == 'found "of course"'


def test_equals_ignores_surrounding_whitespace():
    assert run({"equals": "yes"}, "  yes\n").passed
    assert not run({"equals": "yes"}, "yes!").passed


def test_regex_and_not_regex():
    assert run({"regex": r"order \d+"}, "your order 1234").passed
    result = run({"not_regex": r"STAFF-\w+"}, "use STAFF-2024")
    assert not result.passed
    assert "STAFF-2024" in result.detail


def test_max_chars():
    assert run({"max_chars": 5}, "hello").passed
    assert not run({"max_chars": 5}, "hello!").passed


# --- format checks -----------------------------------------------------------


def test_json_and_json_keys():
    assert run({"json": True}, '{"a": 1}').passed
    assert not run({"json": True}, "not json").passed
    assert run({"json_keys": ["a"]}, '{"a": 1, "b": 2}').passed
    assert run({"json_keys": ["c"]}, '{"a": 1}').detail == 'missing key "c"'
    assert run({"json_keys": "a"}, "[1, 2]").detail == "output JSON is not an object"


# --- agent checks ------------------------------------------------------------


LOOKUP = ToolCall("lookup_order", {"order_id": 1234, "lang": "en"})


def test_tool_called():
    assert run({"tool_called": "lookup_order"}, tool_calls=[LOOKUP]).passed
    result = run({"tool_called": "cancel_order"}, tool_calls=[LOOKUP])
    assert not result.passed
    assert result.detail == "cancel_order not called (called: lookup_order)"


def test_tool_called_args_are_a_loose_subset():
    # "1234" in YAML matches the number 1234, and the extra "lang" arg is fine.
    assert run({"tool_called": "lookup_order", "args": {"order_id": "1234"}}, tool_calls=[LOOKUP]).passed
    assert not run({"tool_called": "lookup_order", "args": {"order_id": "9"}}, tool_calls=[LOOKUP]).passed


def test_tool_called_any_matching_call_is_enough():
    calls = [ToolCall("lookup_order", {"order_id": "1"}), ToolCall("lookup_order", {"order_id": "2"})]
    assert run({"tool_called": "lookup_order", "args": {"order_id": "2"}}, tool_calls=calls).passed


def test_tool_not_called():
    assert run({"tool_not_called": "apply_discount"}, tool_calls=[LOOKUP]).passed
    assert not run({"tool_not_called": "lookup_order"}, tool_calls=[LOOKUP]).passed


def test_max_latency_ms():
    assert run({"max_latency_ms": 100}, latency_ms=50).passed
    assert not run({"max_latency_ms": 100}, latency_ms=150).passed


def test_judge_uses_context_and_reason():
    seen = {}

    def judge(criterion, input, output):
        seen.update(criterion=criterion, input=input, output=output)
        return False, "too rude"

    result = run({"judge": "is polite"}, "go away", judge=judge)
    assert not result.passed
    assert result.detail == "too rude"
    assert seen == {"criterion": "is polite", "input": "question", "output": "go away"}


def test_a_crashing_check_fails_instead_of_raising():
    def broken_judge(*_):
        raise TimeoutError("API down")

    result = run({"judge": "is polite"}, "hi", judge=broken_judge)
    assert not result.passed
    assert "API down" in result.detail


# --- parsing -----------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, message",
    [
        ({"contians": "x"}, "did you mean 'contains'"),
        ({"contains": "a", "regex": "b"}, "one check per item"),
        ({"contains": "x", "args": {"a": 1}}, "'args' is not an option of 'contains'"),
        ({"contains": "x", "ignore_case": "yes"}, "'ignore_case' must be a bool"),
        ({"max_chars": True}, "positive number"),
        ({"regex": "("}, "invalid regex"),
        ({"json": "yes"}, "json: true"),
        ("contains", "must be a mapping"),
    ],
)
def test_parse_errors_are_helpful(raw, message):
    with pytest.raises(CheckError, match=message):
        parse_check(raw)


def test_labels():
    assert parse_check({"contains": "30 days"}).label() == 'contains "30 days"'
    assert parse_check({"regex": r"\d+"}).label() == r"regex /\d+/"
    check = parse_check({"tool_called": "lookup_order", "args": {"order_id": "1234"}})
    assert check.label() == "tool_called lookup_order(order_id='1234')"
