from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Optional

from src.common.database.database_model import PersonGroupMemory
from src.common.logger import get_logger
from src.person_info.memory_schema import (
    MemoryPointDict,
    _memory_to_log_string,
    _remove_lowest_weight_entry,
    _sanitize_memory_points,
)

logger = get_logger("person_info")


@dataclass(slots=True)
class MemoryStore:
    person_id: str
    memory_points: list[MemoryPointDict]
    group_id: Optional[str]
    persist: Callable[[], None]
    restore: Callable[[list[MemoryPointDict]], None]

    def snapshot(self) -> list[MemoryPointDict]:
        return [point.copy() for point in self.memory_points]


def ensure_memory_points_initialized(person) -> None:
    if not isinstance(person.memory_points, list):
        person.memory_points = []
        return

    sanitized = _sanitize_memory_points(person.memory_points)
    if sanitized != person.memory_points:
        person.memory_points = sanitized


def compact_memory_points(person) -> None:
    ensure_memory_points_initialized(person)


def remove_lowest_weight_point(
    person,
    *,
    predicate: Optional[Callable[[MemoryPointDict], bool]] = None,
) -> Optional[MemoryPointDict]:
    if not person.memory_points:
        return None

    removed_point = _remove_lowest_weight_entry(person.memory_points, predicate=predicate)
    if removed_point:
        logger.debug(f"移除记忆点以腾出空间: {_memory_to_log_string(removed_point)}")
    return removed_point


def _persist_group_memory_points(person_id: str, group_id: str, memory_points: list[MemoryPointDict]) -> None:
    record, created = PersonGroupMemory.get_or_create(
        person_id=person_id,
        group_id=group_id,
        defaults={"memory_points": json.dumps(memory_points, ensure_ascii=False)},
    )
    if not created:
        record.memory_points = json.dumps(memory_points, ensure_ascii=False)
        record.save()


def _restore_group_memory_points(person, group_id: str, memory_points: list[MemoryPointDict]) -> None:
    if getattr(person, "_group_memory_cache", None) is None:
        person._group_memory_cache = {}
    person._group_memory_cache[group_id] = memory_points


def get_group_memory_points(person, group_id: str) -> list[MemoryPointDict]:
    group_id = (group_id or "").strip()
    if not group_id:
        return []

    if getattr(person, "_group_memory_cache", None) is None:
        person._group_memory_cache = {}
    if group_id in person._group_memory_cache:
        return person._group_memory_cache[group_id]

    memory_points: list[MemoryPointDict] = []
    try:
        record = PersonGroupMemory.get_or_none(
            (PersonGroupMemory.person_id == person.person_id) & (PersonGroupMemory.group_id == group_id)
        )
        if record and record.memory_points:
            loaded_points = json.loads(record.memory_points)
            if isinstance(loaded_points, list):
                memory_points = _sanitize_memory_points(loaded_points)
    except Exception as exc:
        logger.warning(f"解析用户 {person.person_id} 在群 {group_id} 的记忆失败: {exc}")

    person._group_memory_cache[group_id] = memory_points
    return memory_points


def resolve_global_memory_store(person) -> MemoryStore:
    compact_memory_points(person)
    return MemoryStore(
        person_id=person.person_id,
        memory_points=person.memory_points,
        group_id=None,
        persist=person.sync_to_database,
        restore=lambda previous_points: setattr(person, "memory_points", previous_points),
    )


def resolve_group_memory_store(person, group_id: str) -> Optional[MemoryStore]:
    group_id = (group_id or "").strip()
    if not group_id:
        return None

    memory_points = get_group_memory_points(person, group_id)
    return MemoryStore(
        person_id=person.person_id,
        memory_points=memory_points,
        group_id=group_id,
        persist=lambda: _persist_group_memory_points(person.person_id, group_id, memory_points),
        restore=lambda previous_points: _restore_group_memory_points(person, group_id, previous_points),
    )


__all__ = [
    "MemoryStore",
    "compact_memory_points",
    "ensure_memory_points_initialized",
    "get_group_memory_points",
    "remove_lowest_weight_point",
    "resolve_global_memory_store",
    "resolve_group_memory_store",
]
