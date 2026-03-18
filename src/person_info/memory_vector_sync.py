from __future__ import annotations

import asyncio
from typing import Any

from src.common.vector_store.manager import vector_manager


def schedule_vector_upsert(
    category: str,
    memory_content: str,
    vector_metadata: dict[str, Any],
    removed_ids: list[str],
) -> None:
    if removed_ids:
        asyncio.create_task(vector_manager.delete(removed_ids))
    if category == "关系":
        asyncio.create_task(vector_manager.add_relation(memory_content, vector_metadata))
    else:
        asyncio.create_task(vector_manager.add_memory(memory_content, vector_metadata))


async def await_vector_upsert(
    category: str,
    memory_content: str,
    vector_metadata: dict[str, Any],
    removed_ids: list[str],
) -> None:
    if removed_ids:
        await vector_manager.delete(removed_ids)
    if category == "关系":
        await vector_manager.add_relation(memory_content, vector_metadata)
    else:
        await vector_manager.add_memory(memory_content, vector_metadata)


__all__ = ["await_vector_upsert", "schedule_vector_upsert"]
