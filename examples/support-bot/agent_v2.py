"""Toy customer-support agent for an online shop, version 2.

What changed: a new shipping FAQ that answers in the customer's language.
It fixes the Spanish answers from v1... but its keywords also catch
"Where is my order #1234?", so that question now gets the generic FAQ
instead of an order lookup. That is the regression the suite catches.
"""

import json
import re
import sys

SHIPPING_WORDS = ("ship", "delivery", "where is", "envío", "envio")


def respond(text: str, attempt: int) -> dict:
    lowered = text.lower()
    order = re.search(r"#?(\d{4,})", text)

    # New in v2: shipping FAQ, checked first.
    if any(word in lowered for word in SHIPPING_WORDS):
        if "¿" in text:
            return {"output": "¡Sí! Hacemos envíos a toda España, también a Canarias, en 3-5 días laborables."}
        return {"output": "We ship worldwide. Standard delivery takes 3-5 business days."}

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
    return {"output": "Hi! How can I help you today?"}


if __name__ == "__main__":
    request = json.load(sys.stdin)
    print(json.dumps(respond(request["input"], request.get("attempt", 0)), ensure_ascii=False))
