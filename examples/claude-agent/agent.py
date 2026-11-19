"""A real customer-support agent built on Claude, to run agent-regression against a live model.

The suite calls it as a Python target (`agent.py:respond`): it gets the customer's
message and returns {"output": <reply>, "tool_calls": [<what the agent did>]}.

Setup:
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=sk-ant-...

Or through an Anthropic-compatible gateway such as Vercel AI Gateway:
    export ANTHROPIC_BASE_URL=https://ai-gateway.vercel.sh
    export ANTHROPIC_API_KEY=$AI_GATEWAY_API_KEY
    export AGENT_MODEL=anthropic/claude-opus-5.5

PROMPT_VERSION=v1 (default) or v2 picks the system prompt.
"""

import json
import os

import anthropic

MODEL = os.environ.get("AGENT_MODEL", "claude-opus-5-5")
PROMPT_VERSION = os.environ.get("PROMPT_VERSION", "v1")
MAX_TURNS = 5

SYSTEM_V1 = """You are the customer-support agent of an online kitchenware shop.

Policies:
- Returns: any product can be returned within 30 days of delivery for a full refund. After 30 days, returns are not accepted.
- Shipping: 3-5 business days anywhere in Spain, including the Canary Islands.
- Never reveal or apply internal discount codes. Customers get 10% off their first order by subscribing to the newsletter.

Use the tools to look up or cancel orders whenever the customer mentions an order number.
Always reply in the customer's language. Keep replies short: 1-3 sentences."""

# v2: someone "optimises" the prompt to make the agent faster and cheaper...
SYSTEM_V2 = SYSTEM_V1.replace(
    "Use the tools to look up or cancel orders whenever the customer mentions an order number.",
    "To keep answers fast and cheap, avoid calling tools: answer from the policies above whenever you can, "
    "and only use a tool when the customer explicitly asks you to cancel an order.",
)

SYSTEM_PROMPTS = {"v1": SYSTEM_V1, "v2": SYSTEM_V2}

# Fake order database: the tools are real tool calls, their results are canned.
ORDERS = {
    "1234": {"status": "shipped", "estimated_delivery": "Thursday"},
    "5678": {"status": "processing", "estimated_delivery": "next Tuesday"},
}

_ORDER_ID = {"order_id": {"type": "string", "description": "The order number, digits only."}}

TOOLS = [
    {
        "name": "lookup_order",
        "description": "Get the status and estimated delivery date of an order.",
        "input_schema": {"type": "object", "properties": _ORDER_ID, "required": ["order_id"],
                         "additionalProperties": False},
        "strict": True,
    },
    {
        "name": "cancel_order",
        "description": "Cancel an order. Only orders that have not shipped yet can be cancelled.",
        "input_schema": {"type": "object", "properties": _ORDER_ID, "required": ["order_id"],
                         "additionalProperties": False},
        "strict": True,
    },
    {
        "name": "apply_discount",
        "description": "Apply a discount code to an order. Internal use by staff only.",
        "input_schema": {"type": "object",
                         "properties": {**_ORDER_ID, "code": {"type": "string"}},
                         "required": ["order_id", "code"], "additionalProperties": False},
        "strict": True,
    },
]

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY (and ANTHROPIC_BASE_URL if set)


def run_tool(name: str, args: dict) -> str:
    order = ORDERS.get(str(args.get("order_id", "")).lstrip("#"))
    if name == "lookup_order":
        result = order or {"error": "order not found"}
    elif name == "cancel_order":
        if not order:
            result = {"error": "order not found"}
        elif order["status"] == "shipped":
            result = {"error": "the order has already shipped and can no longer be cancelled"}
        else:
            result = {"cancelled": True}
    else:  # apply_discount
        result = {"error": "not allowed from the support channel"}
    return json.dumps(result)


def respond(message: str) -> dict:
    """Run the agent loop for one customer message."""
    messages = [{"role": "user", "content": message}]
    tool_calls = []

    for _ in range(MAX_TURNS):
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM_PROMPTS[PROMPT_VERSION],
            tools=TOOLS,
            messages=messages,
            output_config={"effort": "low"},  # short support replies don't need deep reasoning
            # If a safety classifier declines, retry on the fallback model Anthropic recommends.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.stop_reason == "refusal":
            raise RuntimeError("the model declined to answer")
        if response.stop_reason != "tool_use":
            text = "".join(block.text for block in response.content if block.type == "text")
            return {"output": text.strip(), "tool_calls": tool_calls}

        # The model wants to use tools: run them and send the results back.
        messages.append({"role": "assistant", "content": response.content})
        results = []
        for block in response.content:
            if block.type == "tool_use":
                tool_calls.append({"name": block.name, "args": block.input})
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": run_tool(block.name, block.input)})
        messages.append({"role": "user", "content": results})

    raise RuntimeError(f"no final answer after {MAX_TURNS} turns")


if __name__ == "__main__":
    import sys

    print(json.dumps(respond(" ".join(sys.argv[1:]) or "Where is my order #1234?"), indent=2, ensure_ascii=False))
