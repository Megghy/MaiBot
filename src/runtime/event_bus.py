from __future__ import annotations

import inspect
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Awaitable, Callable


HandlerResult = Any | tuple[bool, Any]
AsyncHandler = Callable[[Any], Awaitable[HandlerResult]]


@dataclass(slots=True)
class EventHandlerEntry:
    event_type: str
    handler_name: str
    handler: AsyncHandler
    weight: int = 0
    intercept: bool = False


class EventBus:
    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandlerEntry]] = defaultdict(list)

    def register(
        self,
        event_type: str,
        handler_name: str,
        handler: AsyncHandler,
        *,
        weight: int = 0,
        intercept: bool = False,
    ) -> None:
        entries = self._handlers[event_type]
        if any(entry.handler_name == handler_name for entry in entries):
            raise ValueError(f"事件处理器已存在: {handler_name}")
        entries.append(
            EventHandlerEntry(
                event_type=event_type,
                handler_name=handler_name,
                handler=handler,
                weight=weight,
                intercept=intercept,
            )
        )
        entries.sort(key=lambda entry: entry.weight, reverse=True)

    def unregister(self, handler_name: str, event_type: str | None = None) -> bool:
        target_events = [event_type] if event_type is not None else list(self._handlers.keys())
        removed = False
        for current_event in target_events:
            if current_event is None:
                continue
            handlers = self._handlers.get(current_event, [])
            remaining = [entry for entry in handlers if entry.handler_name != handler_name]
            if len(remaining) != len(handlers):
                removed = True
                if remaining:
                    self._handlers[current_event] = remaining
                else:
                    self._handlers.pop(current_event, None)
        return removed

    async def dispatch(self, event_type: str, payload: Any = None) -> tuple[bool, Any]:
        current_payload = payload
        for entry in self._handlers.get(event_type, []):
            result = entry.handler(current_payload)
            if inspect.isawaitable(result):
                result = await result

            continue_flag = True
            if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], bool):
                continue_flag, current_payload = result
            elif result is not None:
                current_payload = result

            if entry.intercept and not continue_flag:
                return False, current_payload
            if not continue_flag:
                return False, current_payload
        return True, current_payload
