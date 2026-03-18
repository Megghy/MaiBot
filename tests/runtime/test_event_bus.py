from __future__ import annotations

import pytest

from src.runtime.event_bus import EventBus


@pytest.mark.asyncio
async def test_event_bus_dispatch_orders_by_weight_and_propagates_payload():
    bus = EventBus()
    seen = []

    async def low(payload):
        seen.append(("low", payload))
        return payload + ["low"]

    async def high(payload):
        seen.append(("high", payload))
        return payload + ["high"]

    bus.register("message.received", "low", low, weight=10)
    bus.register("message.received", "high", high, weight=100)

    continue_flag, payload = await bus.dispatch("message.received", [])

    assert continue_flag is True
    assert payload == ["high", "low"]
    assert seen == [("high", []), ("low", ["high"])]


@pytest.mark.asyncio
async def test_event_bus_dispatch_stops_when_handler_intercepts():
    bus = EventBus()
    seen = []

    async def stop(payload):
        seen.append("stop")
        return False, payload + ["blocked"]

    async def later(payload):
        seen.append("later")
        return payload + ["later"]

    bus.register("message.received", "stop", stop, weight=100, intercept=True)
    bus.register("message.received", "later", later, weight=10)

    continue_flag, payload = await bus.dispatch("message.received", [])

    assert continue_flag is False
    assert payload == ["blocked"]
    assert seen == ["stop"]
