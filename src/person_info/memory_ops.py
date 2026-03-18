from __future__ import annotations

import asyncio
import random
from typing import Any, Callable, Optional

from src.common.logger import get_logger
from src.common.vector_store.manager import vector_manager
import src.person_info.memory_repository as memory_repository
from src.person_info.memory_deduper import (
    NormalizedMemoryWrite,
    build_vector_metadata,
    create_memory_point,
    find_duplicate_point,
    merge_duplicate_point,
    normalize_write_request,
    trim_memory_points,
)
from src.person_info.memory_repository import (
    MemoryStore,
    resolve_global_memory_store,
    resolve_group_memory_store,
)
from src.person_info.memory_schema import (
    MEMORY_MAX_WEIGHT,
    MEMORY_RECENCY_TIMESPAN,
    MemoryPointDict,
    _current_timestamp,
    _memory_to_log_string,
    calculate_string_similarity,
    get_category_from_memory,
    get_memory_content_from_memory,
    get_weight_from_memory,
)
from src.person_info.memory_vector_sync import await_vector_upsert, schedule_vector_upsert

logger = get_logger("person_info")

ensure_memory_points_initialized = memory_repository.ensure_memory_points_initialized
compact_memory_points = memory_repository.compact_memory_points
remove_lowest_weight_point = memory_repository.remove_lowest_weight_point
get_group_memory_points = memory_repository.get_group_memory_points


def _resolve_memory_store(person, group_id: Optional[str]) -> Optional[MemoryStore]:
    if group_id is None:
        return resolve_global_memory_store(person)
    return resolve_group_memory_store(person, group_id)


def _prepare_memory_write(
    person,
    memory_content: str,
    *,
    category: Optional[str],
    weight: float,
    similarity_fn: Optional[Callable[[str, str], float]],
    group_id: Optional[str] = None,
) -> Optional[tuple[MemoryStore, NormalizedMemoryWrite]]:
    write_request = normalize_write_request(
        person,
        memory_content,
        category=category,
        weight=weight,
        similarity_fn=similarity_fn,
        group_id=group_id,
    )
    if not write_request:
        return None

    store = _resolve_memory_store(person, write_request.group_id)
    if not store:
        return None

    return store, write_request


def _apply_memory_write(
    store: MemoryStore,
    write_request: NormalizedMemoryWrite,
    *,
    source: Optional[str],
    context: Optional[str],
) -> tuple[str, str, dict[str, Any], list[str]]:
    now = _current_timestamp()
    vector_metadata = build_vector_metadata(
        store.person_id,
        write_request.category,
        write_request.weight,
        now,
        source=source,
        context=context,
        group_id=store.group_id,
    )
    removed_ids: list[str] = []
    duplicate_point = find_duplicate_point(
        store.memory_points,
        write_request.category,
        write_request.memory_content,
        write_request.similarity_fn,
    )

    if duplicate_point is not None:
        vector_metadata["id"] = merge_duplicate_point(
            duplicate_point,
            write_request.memory_content,
            write_request.weight,
            now,
            source=source,
            context=context,
        )
    else:
        removed_ids = trim_memory_points(store.memory_points, write_request.category)
        new_point = create_memory_point(
            write_request.category,
            write_request.memory_content,
            write_request.weight,
            now,
            source=source,
            context=context,
        )
        store.memory_points.append(new_point)
        vector_metadata["id"] = new_point["id"]
        if store.group_id:
            logger.debug(f"新增群记忆点: {_memory_to_log_string(new_point)}")
        else:
            logger.debug(f"新增记忆点: {_memory_to_log_string(new_point)}")

    return write_request.category, write_request.memory_content, vector_metadata, removed_ids


def _commit_sync_memory_write(
    store: MemoryStore,
    write_request: NormalizedMemoryWrite,
    *,
    source: Optional[str],
    context: Optional[str],
) -> bool:
    category, memory_content, vector_metadata, removed_ids = _apply_memory_write(
        store,
        write_request,
        source=source,
        context=context,
    )
    schedule_vector_upsert(category, memory_content, vector_metadata, removed_ids)
    store.persist()
    return True


async def _commit_async_memory_write(
    store: MemoryStore,
    write_request: NormalizedMemoryWrite,
    *,
    source: Optional[str],
    context: Optional[str],
) -> bool:
    previous_points = store.snapshot()
    category, memory_content, vector_metadata, removed_ids = _apply_memory_write(
        store,
        write_request,
        source=source,
        context=context,
    )

    try:
        await await_vector_upsert(category, memory_content, vector_metadata, removed_ids)
        store.persist()
        return True
    except Exception:
        store.restore(previous_points)
        raise


def delete_memory(person, category: str, memory_content: str, similarity_threshold: float = 0.95) -> int:
    if not person.memory_points:
        return 0

    deleted_count = 0
    memory_points_to_keep: list[MemoryPointDict] = []
    removed_ids: list[str] = []

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


def ensure_memory_capacity(person, category: str) -> None:
    compact_memory_points(person)
    trim_memory_points(person.memory_points, category)


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
    prepared = _prepare_memory_write(
        person,
        memory_content,
        category=category,
        weight=weight,
        similarity_fn=similarity_fn,
    )
    if not prepared:
        return False

    store, write_request = prepared
    return _commit_sync_memory_write(store, write_request, source=source, context=context)


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
    prepared = _prepare_memory_write(
        person,
        memory_content,
        category=category,
        weight=weight,
        similarity_fn=similarity_fn,
    )
    if not prepared:
        return False

    store, write_request = prepared
    return await _commit_async_memory_write(store, write_request, source=source, context=context)


def get_all_category(person) -> list[str]:
    category_list: list[str] = []
    for memory in person.memory_points:
        if not memory:
            continue
        category = get_category_from_memory(memory)
        if category and category not in category_list:
            category_list.append(category)
    return category_list


def get_memory_list_by_category(person, category: str) -> list[MemoryPointDict]:
    memory_list: list[MemoryPointDict] = []
    for memory in person.memory_points:
        if memory and get_category_from_memory(memory) == category:
            memory_list.append(memory)
    return memory_list


def get_random_memory_by_category(person, category: str, num: int = 1) -> list[MemoryPointDict]:
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


def get_relevant_memories_by_category(person, category: str, chat_content: str, num: int = 1) -> list[MemoryPointDict]:
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
) -> list[MemoryPointDict]:
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

    matched_memories: list[MemoryPointDict] = []
    for item in top_results:
        vid = str(item.get("id")) if item.get("id") is not None else None
        if vid and vid in id_to_memory:
            matched_memories.append(id_to_memory[vid])

    if not matched_memories:
        return get_relevant_memories_by_category(person, category, query_text, num)
    return matched_memories


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
    prepared = _prepare_memory_write(
        person,
        memory_content,
        category=category,
        weight=weight,
        similarity_fn=similarity_fn,
        group_id=group_id,
    )
    if not prepared:
        return False

    store, write_request = prepared
    return _commit_sync_memory_write(store, write_request, source=source, context=context)


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
    prepared = _prepare_memory_write(
        person,
        memory_content,
        category=category,
        weight=weight,
        similarity_fn=similarity_fn,
        group_id=group_id,
    )
    if not prepared:
        return False

    store, write_request = prepared
    return await _commit_async_memory_write(store, write_request, source=source, context=context)


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
