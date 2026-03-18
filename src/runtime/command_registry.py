from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable


Matcher = Callable[..., dict[str, Any] | None]
Executor = Callable[..., Awaitable[Any]]


@dataclass(slots=True)
class CommandHandlerEntry:
    name: str
    matcher: Matcher
    executor: Executor
    priority: int = 0


class CommandRegistry:
    def __init__(self) -> None:
        self._entries: list[CommandHandlerEntry] = []

    def register(self, name: str, matcher: Matcher, executor: Executor, *, priority: int = 0) -> None:
        if any(entry.name == name for entry in self._entries):
            raise ValueError(f"命令处理器已存在: {name}")
        self._entries.append(CommandHandlerEntry(name=name, matcher=matcher, executor=executor, priority=priority))
        self._entries.sort(key=lambda entry: entry.priority, reverse=True)

    async def execute(self, message, *, session=None):
        text = getattr(message, "processed_plain_text", "") or ""
        for entry in self._entries:
            match_data = entry.matcher(text, message=message, session=session)
            if match_data is None:
                continue
            if not isinstance(match_data, dict):
                raise TypeError(f"命令匹配器 {entry.name} 必须返回 dict 或 None")
            return await entry.executor(message, session=session, **match_data)
        return None
