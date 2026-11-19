"""Toy customer-support agent for an online shop, version 1.

A stand-in for a real LLM agent so the demo runs offline and for free.
It speaks the agent-regression command protocol: one JSON object on stdin,
one JSON object on stdout with the answer and the tools it called.

Known issue in v1: Spanish questions only get a Spanish answer on the first
attempt; afterwards it drifts to English, like a real model sometimes does.
"""

import json
import re
import sys


def respond(text: str, attempt: int) -> dict:
    lowered = text.lower()
    order = re.search(r"#?(\d{4,})", text)

    if "cancel" in lowered and order:
        order_id = order.group(1)
        return {
            "output": f"Done: order {order_id} is cancelled. You'll get a confirmation email shortly.",
            "tool_calls": [{"name": "cancel_order", "args": {"order_id": order_id}}],
        }
    if order:
        order_id = order.group(1)
        return {
            "output": f"Order {order_id} is on its way and should arrive on Thursday.",
            "tool_calls": [{"name": "lookup_order", "args": {"order_id": order_id}}],
        }
    if "return" in lowered or "refund" in lowered:
        return {"output": "You can return any product within 30 days of delivery for a full refund."}
    if "discount" in lowered or "code" in lowered:
        return {"output": "I can't share internal codes, but our newsletter gives you 10% off your first order."}
    if "¿" in text:
        if attempt == 0:
            return {"output": "¡Sí! Hacemos envíos a toda España, también a Canarias, en 3-5 días."}
        return {"output": "Yes! We ship to all of Spain, including the Canary Islands, in 3-5 days."}
    return {"output": "Hi! How can I help you today?"}


if __name__ == "__main__":
    request = json.load(sys.stdin)
    print(json.dumps(respond(request["input"], request.get("attempt", 0)), ensure_ascii=False))
