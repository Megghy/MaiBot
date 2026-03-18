import asyncio
import json
import random
import uuid
from typing import Any, Callable, List, Optional

from src.common.database.database_model import PersonGroupMemory
from src.common.logger import get_logger
from src.common.vector_store.manager import vector_manager
from src.person_info.memory_schema import (
    MAX_MEMORY_POINTS,
    MAX_MEMORY_POINTS_PER_CATEGORY,
    MEMORY_MAX_WEIGHT,
    MEMORY_RECENCY_TIMESPAN,
    MEMORY_SIMILARITY_THRESHOLD,
    MemoryPointDict,
    _current_timestamp,
    _memory_to_log_string,
    _normalize_category,
    _normalize_weight,
    _remove_lowest_weight_entry,
    _sanitize_memory_points,
    calculate_string_similarity,
    get_category_from_memory,
    get_memory_content_from_memory,
    get_weight_from_memory,
    infer_memory_category,
)

logger = get_logger("person_info")


def _schedule_vector_upsert(
    category: str, memory_content: str, vector_metadata: dict[str, Any], removed_ids: List[str]
) -> None:
    if removed_ids:
        asyncio.create_task(vector_manager.delete(removed_ids))
    if category == "关系":
        asyncio.create_task(vector_manager.add_relation(memory_content, vector_metadata))
    else:
        asyncio.create_task(vector_manager.add_memory(memory_content, vector_metadata))


async def _await_vector_upsert(
    category: str,
    memory_content: str,
    vector_metadata: dict[str, Any],
    removed_ids: List[str],
) -> None:
    if removed_ids:
        await vector_manager.delete(removed_ids)
    if category == "关系":
        await vector_manager.add_relation(memory_content, vector_metadata)
    else:
        await vector_manager.add_memory(memory_content, vector_metadata)


def _normalize_write_request(
    person,
    memory_content: str,
    *,
    category: Optional[str],
    weight: float,
    similarity_fn: Optional[Callable[[str, str], float]],
    group_id: Optional[str] = None,
) -> Optional[tuple[Optional[str], str, str, float, Callable[[str, str], float]]]:
    if not person.is_known:
        logger.debug(f"用户 {person.person_id} 尚未被标记为已认识，无法添加记忆")
        return None

    normalized_group_id = (group_id or "").strip() if group_id is not None else None
    if group_id is not None and not normalized_group_id:
        return None

    normalized_content = (memory_content or "").strip()
    if not normalized_content:
        return None

    return (
        normalized_group_id,
        normalized_content,
        _normalize_category(category or infer_memory_category(normalized_content)),
        _normalize_weight(weight),
        similarity_fn or calculate_string_similarity,
    )


def _find_duplicate_point(
    memory_points: List[MemoryPointDict],
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


def _build_vector_metadata(
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


def _merge_duplicate_point(
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


def _create_memory_point(
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


def _trim_memory_points(memory_points: List[MemoryPointDict], category: str) -> List[str]:
    removed_ids: List[str] = []
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


def _persist_group_memory_points(person_id: str, group_id: str, memory_points: List[MemoryPointDict]) -> None:
    record, created = PersonGroupMemory.get_or_create(
        person_id=person_id,
        group_id=group_id,
        defaults={"memory_points": json.dumps(memory_points, ensure_ascii=False)},
    )
    if not created:
        record.memory_points = json.dumps(memory_points, ensure_ascii=False)
        record.save()


def delete_memory(person, category: str, memory_content: str, similarity_threshold: float = 0.95) -> int:
    if not person.memory_points:
        return 0

    deleted_count = 0
    memory_points_to_keep: List[MemoryPointDict] = []
    removed_ids: List[str] = []

    for memory_point in person.memory_points:
        if not memory_point:
            continue
        memory_category = memory_point.get("category")
        memory_text = memory_point.get("content") or ""
        if memory_category != category:
            memory_points_to_keep.append(memory_point)
            continue

        similarity = calculate_string_similarity(memory_content, memory_text)
        if similarity >= similarity_threshold:
            deleted_count += 1
            if memory_point.get("id"):
                removed_ids.append(str(memory_point["id"]))
            logger.debug(f"删除记忆点: {_memory_to_log_string(memory_point)} (相似度: {similarity:.4f})")
        else:
            memory_points_to_keep.append(memory_point)

    person.memory_points = memory_points_to_keep

    if deleted_count > 0:
        if removed_ids:
            asyncio.create_task(vector_manager.delete(removed_ids))
        person.sync_to_database()
        logger.info(f"成功删除 {deleted_count} 个记忆点，分类: {category}")
    return deleted_count


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


def ensure_memory_capacity(person, category: str) -> None:
    compact_memory_points(person)
    _trim_memory_points(person.memory_points, category)


def add_memory_point(
    person,
    memory_content: str,
    *,
    category: Optional[str] = None,
    weight: float = 1.0,
    similarity_fn: Optional[Callable[[str, str], float]] = None,
    source: Optional[str] = None,
    context: Optional[str] = None,
) -> bool:
    normalized = _normalize_write_request(
        person,
        memory_content,
        category=category,
        weight=weight,
        similarity_fn=similarity_fn,
    )
    if not normalized:
        return False

    _, memory_content, category, weight_value, similarity_fn = normalized
    compact_memory_points(person)

    now = _current_timestamp()
    vector_metadata = _build_vector_metadata(
        person.person_id,
        category,
        weight_value,
        now,
        source=source,
        context=context,
    )
    removed_ids: List[str] = []
    duplicate_point = _find_duplicate_point(person.memory_points, category, memory_content, similarity_fn)

    if duplicate_point is not None:
        vector_metadata["id"] = _merge_duplicate_point(
            duplicate_point,
            memory_content,
            weight_value,
            now,
            source=source,
            context=context,
        )
    else:
        removed_ids = _trim_memory_points(person.memory_points, category)
        new_point = _create_memory_point(
            category,
            memory_content,
            weight_value,
            now,
            source=source,
            context=context,
        )
        person.memory_points.append(new_point)
        vector_metadata["id"] = new_point["id"]
        logger.debug(f"新增记忆点: {_memory_to_log_string(new_point)}")

    _schedule_vector_upsert(category, memory_content, vector_metadata, removed_ids)
    person.sync_to_database()
    return True


async def async_add_memory_point(
    person,
    memory_content: str,
    *,
    category: Optional[str] = None,
    weight: float = 1.0,
    similarity_fn: Optional[Callable[[str, str], float]] = None,
    source: Optional[str] = None,
    context: Optional[str] = None,
) -> bool:
    normalized = _normalize_write_request(
        person,
        memory_content,
        category=category,
        weight=weight,
        similarity_fn=similarity_fn,
    )
    if not normalized:
        return False

    _, memory_content, category, weight_value, similarity_fn = normalized
    compact_memory_points(person)
    previous_points = [point.copy() for point in person.memory_points]

    now = _current_timestamp()
    vector_metadata = _build_vector_metadata(
        person.person_id,
        category,
        weight_value,
        now,
        source=source,
        context=context,
    )
    removed_ids: List[str] = []
    duplicate_point = _find_duplicate_point(person.memory_points, category, memory_content, similarity_fn)

    if duplicate_point is not None:
        vector_metadata["id"] = _merge_duplicate_point(
            duplicate_point,
            memory_content,
            weight_value,
            now,
            source=source,
            context=context,
        )
    else:
        removed_ids = _trim_memory_points(person.memory_points, category)
        new_point = _create_memory_point(
            category,
            memory_content,
            weight_value,
            now,
            source=source,
            context=context,
        )
        person.memory_points.append(new_point)
        vector_metadata["id"] = new_point["id"]
        logger.debug(f"新增记忆点: {_memory_to_log_string(new_point)}")

    try:
        await _await_vector_upsert(category, memory_content, vector_metadata, removed_ids)
        person.sync_to_database()
        return True
    except Exception:
        person.memory_points = previous_points
        raise


def get_all_category(person) -> List[str]:
    category_list: List[str] = []
    for memory in person.memory_points:
        if not memory:
            continue
        category = get_category_from_memory(memory)
        if category and category not in category_list:
            category_list.append(category)
    return category_list


def get_memory_list_by_category(person, category: str) -> List[MemoryPointDict]:
    memory_list: List[MemoryPointDict] = []
    for memory in person.memory_points:
        if memory and get_category_from_memory(memory) == category:
            memory_list.append(memory)
    return memory_list


def get_random_memory_by_category(person, category: str, num: int = 1) -> List[MemoryPointDict]:
    memory_list = get_memory_list_by_category(person, category)
    if len(memory_list) <= num:
        return memory_list
    return random.sample(memory_list, num)


def calculate_relevance(memory_content: str, chat_content: str) -> float:
    if not memory_content or not chat_content:
        return 0.0

    def get_bigrams(text: str) -> set[str]:
        return {text[i : i + 2] for i in range(len(text) - 1)}

    mem_grams = get_bigrams(memory_content)
    chat_grams = get_bigrams(chat_content)
    if not mem_grams:
        return 0.0
    return len(mem_grams & chat_grams) / len(mem_grams)


def get_relevant_memories_by_category(person, category: str, chat_content: str, num: int = 1) -> List[MemoryPointDict]:
    memory_list = get_memory_list_by_category(person, category)
    if not memory_list:
        return []
    if not chat_content:
        return get_random_memory_by_category(person, category, num)

    scored_memories = []
    now = _current_timestamp()
    for memory in memory_list:
        relevance = calculate_relevance(get_memory_content_from_memory(memory), chat_content)
        weight_bonus = get_weight_from_memory(memory) / MEMORY_MAX_WEIGHT
        last_used = memory.get("last_used_at") or memory.get("updated_at") or memory.get("created_at") or now
        recency = max(0.0, 1.0 - (now - last_used) / MEMORY_RECENCY_TIMESPAN)
        score = relevance * 0.6 + weight_bonus * 0.25 + recency * 0.15
        scored_memories.append((score, memory))

    scored_memories.sort(key=lambda item: item[0], reverse=True)
    if scored_memories and scored_memories[0][0] == 0:
        return memory_list if len(memory_list) < num else random.sample(memory_list, num)
    return [memory for _, memory in scored_memories[:num]]


async def get_relevant_memories_by_category_vector(
    person,
    category: str,
    query_text: str,
    num: int = 1,
) -> List[MemoryPointDict]:
    query_text = (query_text or "").strip()
    if not query_text or not person.is_known:
        return get_relevant_memories_by_category(person, category, query_text, num)

    def _filter(meta: dict[str, Any]) -> bool:
        return meta.get("person_id") == person.person_id and meta.get("category") == category

    try:
        vector_results = await vector_manager.search_memory(query_text, k=max(num * 2, num + 1), filter_fn=_filter)
    except Exception as exc:
        logger.warning(f"向量关系记忆检索失败, 使用回退逻辑: {exc}")
        return get_relevant_memories_by_category(person, category, query_text, num)

    if not vector_results:
        return get_relevant_memories_by_category(person, category, query_text, num)

    top_results = sorted(vector_results, key=lambda item: float(item.get("similarity", 0.0)), reverse=True)[:num]
    id_to_memory: dict[str, MemoryPointDict] = {}
    for memory in person.memory_points:
        if memory and memory.get("id") is not None:
            id_to_memory[str(memory["id"])] = memory

    matched_memories: List[MemoryPointDict] = []
    for item in top_results:
        vid = str(item.get("id")) if item.get("id") is not None else None
        if vid and vid in id_to_memory:
            matched_memories.append(id_to_memory[vid])

    if not matched_memories:
        return get_relevant_memories_by_category(person, category, query_text, num)
    return matched_memories


def get_group_memory_points(person, group_id: str) -> List[MemoryPointDict]:
    group_id = (group_id or "").strip()
    if not group_id:
        return []

    if getattr(person, "_group_memory_cache", None) is None:
        person._group_memory_cache = {}
    if group_id in person._group_memory_cache:
        return person._group_memory_cache[group_id]

    memory_points: List[MemoryPointDict] = []
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


def add_group_memory_point(
    person,
    group_id: str,
    memory_content: str,
    *,
    category: Optional[str] = None,
    weight: float = 1.0,
    similarity_fn: Optional[Callable[[str, str], float]] = None,
    source: Optional[str] = None,
    context: Optional[str] = None,
) -> bool:
    normalized = _normalize_write_request(
        person,
        memory_content,
        category=category,
        weight=weight,
        similarity_fn=similarity_fn,
        group_id=group_id,
    )
    if not normalized:
        return False

    group_id, memory_content, category, weight_value, similarity_fn = normalized
    memory_points = get_group_memory_points(person, group_id)

    now = _current_timestamp()
    vector_metadata = _build_vector_metadata(
        person.person_id,
        category,
        weight_value,
        now,
        source=source,
        context=context,
        group_id=group_id,
    )
    removed_ids: List[str] = []
    duplicate_point = _find_duplicate_point(memory_points, category, memory_content, similarity_fn)

    if duplicate_point is not None:
        vector_metadata["id"] = _merge_duplicate_point(
            duplicate_point,
            memory_content,
            weight_value,
            now,
            source=source,
            context=context,
        )
    else:
        removed_ids = _trim_memory_points(memory_points, category)
        new_point = _create_memory_point(
            category,
            memory_content,
            weight_value,
            now,
            source=source,
            context=context,
        )
        memory_points.append(new_point)
        vector_metadata["id"] = new_point["id"]
        logger.debug(f"新增群记忆点: {_memory_to_log_string(new_point)}")

    _schedule_vector_upsert(category, memory_content, vector_metadata, removed_ids)
    try:
        _persist_group_memory_points(person.person_id, group_id, memory_points)
        return True
    except Exception as exc:
        logger.error(f"同步用户 {person.person_id} 在群 {group_id} 的群记忆到数据库时出错: {exc}")
        return False


async def async_add_group_memory_point(
    person,
    group_id: str,
    memory_content: str,
    *,
    category: Optional[str] = None,
    weight: float = 1.0,
    similarity_fn: Optional[Callable[[str, str], float]] = None,
    source: Optional[str] = None,
    context: Optional[str] = None,
) -> bool:
    normalized = _normalize_write_request(
        person,
        memory_content,
        category=category,
        weight=weight,
        similarity_fn=similarity_fn,
        group_id=group_id,
    )
    if not normalized:
        return False

    group_id, memory_content, category, weight_value, similarity_fn = normalized
    memory_points = get_group_memory_points(person, group_id)
    previous_points = [point.copy() for point in memory_points]

    now = _current_timestamp()
    vector_metadata = _build_vector_metadata(
        person.person_id,
        category,
        weight_value,
        now,
        source=source,
        context=context,
        group_id=group_id,
    )
    removed_ids: List[str] = []
    duplicate_point = _find_duplicate_point(memory_points, category, memory_content, similarity_fn)

    if duplicate_point is not None:
        vector_metadata["id"] = _merge_duplicate_point(
            duplicate_point,
            memory_content,
            weight_value,
            now,
            source=source,
            context=context,
        )
    else:
        removed_ids = _trim_memory_points(memory_points, category)
        new_point = _create_memory_point(
            category,
            memory_content,
            weight_value,
            now,
            source=source,
            context=context,
        )
        memory_points.append(new_point)
        vector_metadata["id"] = new_point["id"]
        logger.debug(f"新增群记忆点: {_memory_to_log_string(new_point)}")

    try:
        await _await_vector_upsert(category, memory_content, vector_metadata, removed_ids)
        _persist_group_memory_points(person.person_id, group_id, memory_points)
        return True
    except Exception:
        if getattr(person, "_group_memory_cache", None) is None:
            person._group_memory_cache = {}
        person._group_memory_cache[group_id] = previous_points
        raise


def add_group_nick_name(person, group_id: str, group_nick_name: str) -> None:
    if not group_id or not group_nick_name:
        return

    for item in person.group_nick_name:
        if item.get("group_id") == group_id:
            item["group_nick_name"] = group_nick_name
            person.sync_to_database()
            logger.debug(f"更新用户 {person.person_id} 在群 {group_id} 的群昵称为 {group_nick_name}")
            return

    person.group_nick_name.append({"group_id": group_id, "group_nick_name": group_nick_name})
    person.sync_to_database()
    logger.debug(f"添加用户 {person.person_id} 在群 {group_id} 的群昵称 {group_nick_name}")
