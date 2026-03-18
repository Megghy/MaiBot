from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Callable, Optional

from src.common.logger import get_logger
from src.person_info.memory_schema import (
    MEMORY_SIMILARITY_THRESHOLD,
    MAX_MEMORY_POINTS,
    MAX_MEMORY_POINTS_PER_CATEGORY,
    MemoryPointDict,
    _memory_to_log_string,
    _normalize_category,
    _normalize_weight,
    _remove_lowest_weight_entry,
    calculate_string_similarity,
    get_category_from_memory,
    get_memory_content_from_memory,
    infer_memory_category,
)

logger = get_logger("person_info")


@dataclass(slots=True)
class NormalizedMemoryWrite:
    group_id: Optional[str]
    memory_content: str
    category: str
    weight: float
    similarity_fn: Callable[[str, str], float]


def normalize_write_request(
    person,
    memory_content: str,
    *,
    category: Optional[str],
    weight: float,
    similarity_fn: Optional[Callable[[str, str], float]],
    group_id: Optional[str] = None,
) -> Optional[NormalizedMemoryWrite]:
    if not person.is_known:
        logger.debug(f"用户 {person.person_id} 尚未被标记为已认识，无法添加记忆")
        return None

    normalized_group_id = (group_id or "").strip() if group_id is not None else None
    if group_id is not None and not normalized_group_id:
        return None

    normalized_content = (memory_content or "").strip()
    if not normalized_content:
        return None

    return NormalizedMemoryWrite(
        group_id=normalized_group_id,
        memory_content=normalized_content,
        category=_normalize_category(category or infer_memory_category(normalized_content)),
        weight=_normalize_weight(weight),
        similarity_fn=similarity_fn or calculate_string_similarity,
    )


def find_duplicate_point(
    memory_points: list[MemoryPointDict],
    category: str,
    memory_content: str,
    similarity_fn: Callable[[str, str], float],
) -> Optional[MemoryPointDict]:
    for point in memory_points:
        if get_category_from_memory(point) != category:
            continue
        similarity = similarity_fn(get_memory_content_from_memory(point), memory_content)
        if similarity >= MEMORY_SIMILARITY_THRESHOLD:
            return point
    return None


def build_vector_metadata(
    person_id: str,
    category: str,
    weight_value: float,
    now: float,
    *,
    source: Optional[str],
    context: Optional[str],
    group_id: Optional[str] = None,
) -> dict[str, Any]:
    metadata = {
        "person_id": person_id,
        "category": category,
        "weight": weight_value,
        "source": source,
        "context": context,
        "created_at": now,
    }
    if group_id:
        metadata["group_id"] = group_id
        metadata["is_group"] = True
    return metadata


def merge_duplicate_point(
    point: MemoryPointDict,
    memory_content: str,
    weight_value: float,
    now: float,
    *,
    source: Optional[str],
    context: Optional[str],
) -> str:
    old_weight = point.get("weight", 1.0)
    point["weight"] = _normalize_weight(old_weight * 0.7 + weight_value * 0.3 + 0.2)
    point["content"] = memory_content
    point["updated_at"] = now
    point["last_used_at"] = now
    point["hits"] = int(point.get("hits", 0)) + 1
    if source:
        point["source"] = source
    if context:
        point["context"] = context
    if not point.get("id"):
        point["id"] = str(uuid.uuid4())
    logger.debug(f"更新已存在的记忆点: {_memory_to_log_string(point)}")
    return str(point["id"])


def create_memory_point(
    category: str,
    memory_content: str,
    weight_value: float,
    now: float,
    *,
    source: Optional[str],
    context: Optional[str],
) -> MemoryPointDict:
    point: MemoryPointDict = {
        "id": str(uuid.uuid4()),
        "category": category,
        "content": memory_content,
        "weight": weight_value,
        "created_at": now,
        "updated_at": now,
        "last_used_at": now,
        "hits": 1,
    }
    if source:
        point["source"] = source
    if context:
        point["context"] = context
    return point


def trim_memory_points(memory_points: list[MemoryPointDict], category: str) -> list[str]:
    removed_ids: list[str] = []
    category_points = [point for point in memory_points if get_category_from_memory(point) == category]
    if len(category_points) >= MAX_MEMORY_POINTS_PER_CATEGORY:
        removed_point = _remove_lowest_weight_entry(
            memory_points,
            predicate=lambda point: get_category_from_memory(point) == category,
        )
        if removed_point:
            logger.debug(f"移除记忆点以腾出空间: {_memory_to_log_string(removed_point)}")
            if removed_point.get("id"):
                removed_ids.append(str(removed_point["id"]))

    if len(memory_points) >= MAX_MEMORY_POINTS:
        removed_point = _remove_lowest_weight_entry(memory_points)
        if removed_point:
            logger.debug(f"移除记忆点以腾出空间: {_memory_to_log_string(removed_point)}")
            if removed_point.get("id"):
                removed_ids.append(str(removed_point["id"]))

    return removed_ids


__all__ = [
    "NormalizedMemoryWrite",
    "build_vector_metadata",
    "create_memory_point",
    "find_duplicate_point",
    "merge_duplicate_point",
    "normalize_write_request",
    "trim_memory_points",
]
