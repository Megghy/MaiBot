from __future__ import annotations

import json

import pytest

import src.person_info.memory_ops as memory_ops
from src.person_info.memory_repository import resolve_global_memory_store, resolve_group_memory_store


def test_resolve_global_memory_store_initializes_and_persists():
    class DummyPerson:
        def __init__(self) -> None:
            self.person_id = "person-1"
            self.memory_points = None
            self.sync_calls = 0

        def sync_to_database(self) -> None:
            self.sync_calls += 1

    person = DummyPerson()

    store = resolve_global_memory_store(person)
    assert store.group_id is None
    assert person.memory_points == []

    store.memory_points.append({"category": "喜好", "content": "喜欢咖啡", "weight": 1.0})
    store.persist()
    assert person.sync_calls == 1

    restored_points = [{"category": "经历", "content": "去过上海", "weight": 0.8}]
    store.restore(restored_points)
    assert person.memory_points == restored_points


def test_resolve_group_memory_store_loads_persists_and_restores_cache(monkeypatch):
    load_record = type(
        "LoadRecord",
        (),
        {
            "memory_points": json.dumps(
                [{"category": "喜好", "content": "喜欢奶茶", "weight": 1.0}], ensure_ascii=False
            ),
        },
    )()
    persist_record = type(
        "PersistRecord", (), {"memory_points": "", "saved": False, "save": lambda self: setattr(self, "saved", True)}
    )()

    monkeypatch.setattr(
        "src.person_info.memory_repository.PersonGroupMemory.get_or_none",
        lambda *args, **kwargs: load_record,
    )
    monkeypatch.setattr(
        "src.person_info.memory_repository.PersonGroupMemory.get_or_create",
        lambda *args, **kwargs: (persist_record, False),
    )

    class DummyPerson:
        def __init__(self) -> None:
            self.person_id = "person-2"
            self._group_memory_cache = {}

    person = DummyPerson()

    store = resolve_group_memory_store(person, "group-1")
    assert store is not None
    assert store.group_id == "group-1"
    assert len(store.memory_points) == 1
    assert store.memory_points[0]["content"] == "喜欢奶茶"

    store.memory_points.append({"category": "经历", "content": "群里常聊游戏", "weight": 0.6})
    store.persist()

    saved_points = json.loads(persist_record.memory_points)
    assert persist_record.saved is True
    assert len(saved_points) == 2
    assert saved_points[-1]["content"] == "群里常聊游戏"

    restored_points = [{"category": "性格", "content": "比较外向", "weight": 1.2}]
    store.restore(restored_points)
    assert person._group_memory_cache["group-1"] == restored_points


def test_add_memory_point_uses_shared_store_pipeline(monkeypatch):
    scheduled_calls = []

    monkeypatch.setattr(
        memory_ops,
        "schedule_vector_upsert",
        lambda category, memory_content, vector_metadata, removed_ids: scheduled_calls.append(
            (category, memory_content, vector_metadata, removed_ids)
        ),
    )

    class DummyPerson:
        def __init__(self) -> None:
            self.person_id = "person-3"
            self.is_known = True
            self.memory_points = []
            self.sync_calls = 0

        def sync_to_database(self) -> None:
            self.sync_calls += 1

    person = DummyPerson()

    added = memory_ops.add_memory_point(person, "喜欢热拿铁", category="喜好", source="unit-test")

    assert added is True
    assert person.sync_calls == 1
    assert len(person.memory_points) == 1
    assert person.memory_points[0]["content"] == "喜欢热拿铁"
    assert scheduled_calls[0][0] == "喜好"
    assert scheduled_calls[0][1] == "喜欢热拿铁"
    assert scheduled_calls[0][2]["person_id"] == "person-3"


@pytest.mark.asyncio
async def test_async_add_group_memory_point_restores_previous_cache_on_failure(monkeypatch):
    original_points = [{"category": "经历", "content": "之前一起看电影", "weight": 0.9}]

    async def fake_await_vector_upsert(*args, **kwargs):
        raise RuntimeError("vector boom")

    monkeypatch.setattr(memory_ops, "await_vector_upsert", fake_await_vector_upsert)

    class DummyPerson:
        def __init__(self) -> None:
            self.person_id = "person-4"
            self.is_known = True
            self._group_memory_cache = {"group-1": [point.copy() for point in original_points]}

        def sync_to_database(self) -> None:
            raise AssertionError("group memory write should not touch global sync")

    person = DummyPerson()

    with pytest.raises(RuntimeError, match="vector boom"):
        await memory_ops.async_add_group_memory_point(person, "group-1", "最近在群里聊摄影", category="经历")

    assert person._group_memory_cache["group-1"] == original_points
