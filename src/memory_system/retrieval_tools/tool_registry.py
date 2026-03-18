from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from src.common.logger import get_logger

logger = get_logger("memory_retrieval_tools")


@dataclass(slots=True)
class MemoryToolSpec:
    name: str
    description: str
    parameters: list[dict[str, Any]]
    execute_func: Callable[..., Awaitable[str]]

    async def execute(self, **kwargs) -> str:
        return await self.execute_func(**kwargs)


_tool_specs: dict[str, MemoryToolSpec] = {}


def register_memory_retrieval_tool(
    name: str,
    description: str,
    parameters: list[dict[str, Any]],
    execute_func: Callable[..., Awaitable[str]],
) -> None:
    if name in _tool_specs:
        logger.debug(f"记忆检索工具 {name} 已存在，跳过重复注册")
        return
    _tool_specs[name] = MemoryToolSpec(
        name=name,
        description=description,
        parameters=list(parameters),
        execute_func=execute_func,
    )
    logger.info(f"注册记忆检索工具: {name}")


def get_registered_memory_tools() -> dict[str, MemoryToolSpec]:
    return _tool_specs.copy()


def get_registered_memory_tool(name: str) -> MemoryToolSpec | None:
    return _tool_specs.get(name)


def clear_registered_memory_tools() -> None:
    _tool_specs.clear()


__all__ = [
    "MemoryToolSpec",
    "clear_registered_memory_tools",
    "get_registered_memory_tool",
    "get_registered_memory_tools",
    "register_memory_retrieval_tool",
]
