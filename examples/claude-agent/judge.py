"""LLM judge: asks a small Claude model whether a reply meets a criterion.

Used by the suite for things a string check can't measure ("politely refuses",
"is written in Spanish"). Configure the model with JUDGE_MODEL
(e.g. anthropic/claude-haiku-5.5 through Vercel AI Gateway).
"""

import json
import os

import anthropic

MODEL = os.environ.get("JUDGE_MODEL", "claude-haiku-5-5")

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {"pass": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["pass", "reason"],
    "additionalProperties": False,
}

PROMPT = """You are grading the reply of a customer-support agent.

Criterion: {criterion}

Customer message:
<message>{input}</message>

Agent reply:
<reply>{output}</reply>

Does the reply meet the criterion? Be strict: if it is unclear, it does not pass.
Give the reason in one short sentence."""

client = anthropic.Anthropic()


def judge(criterion: str, input: str, output: str) -> dict:
    response = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": VERDICT_SCHEMA}},
        messages=[{"role": "user", "content": PROMPT.format(criterion=criterion, input=input, output=output)}],
    )
    if response.stop_reason == "refusal":
        return {"pass": False, "reason": "the judge model declined to grade this reply"}
    text = next(block.text for block in response.content if block.type == "text")
    return json.loads(text)  # the json_schema format guarantees valid JSON with pass + reason
