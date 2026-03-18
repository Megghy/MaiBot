from __future__ import annotations

import pytest

from src.person_info import memory_ops, relationship_ops
from src.person_info.person_info import Person
from src.person_info.person_persistence import load_person_from_database


def test_person_facade_binds_core_memory_and_relation_methods():
    assert Person.add_memory_point is memory_ops.add_memory_point
    assert Person.async_add_memory_point is memory_ops.async_add_memory_point
    assert Person.get_group_memory_points is memory_ops.get_group_memory_points
    assert Person.load_from_database is load_person_from_database
    assert Person.build_relationship is relationship_ops.build_relationship
    assert Person.record_relation is relationship_ops.record_relation


def test_remove_lowest_weight_point_uses_memory_point_predicate():
    class DummyPerson:
        def __init__(self) -> None:
            self.memory_points = [
                {"category": "喜好", "content": "咖啡", "weight": 0.9},
                {"category": "经历", "content": "去了上海", "weight": 0.2},
                {"category": "经历", "content": "喜欢摄影", "weight": 0.6},
            ]

    person = DummyPerson()

    removed = memory_ops.remove_lowest_weight_point(
        person,
        predicate=lambda point: point["category"] == "经历",
    )

    assert removed == {"category": "经历", "content": "去了上海", "weight": 0.2}
    assert person.memory_points == [
        {"category": "喜好", "content": "咖啡", "weight": 0.9},
        {"category": "经历", "content": "喜欢摄影", "weight": 0.6},
    ]


@pytest.mark.asyncio
async def test_build_relationship_handles_empty_random_memory_result():
    class DummyPerson:
        is_known = True
        person_name = "测试用户"
        nickname = "测试用户"
        platform = "qq"
        relation_status = None

        def get_all_category(self):
            return ["喜好"]

        def get_random_memory_by_category(self, category: str, num: int = 1):
            return []

    result = await relationship_ops.build_relationship(DummyPerson())

    assert result == ""


def test_load_person_from_database_keeps_empty_memory_points(monkeypatch):
    class FakeRecord:
        user_id = "user-1"
        platform = "qq"
        is_known = True
        nickname = "昵称"
        person_name = "名字"
        name_reason = None
        know_times = 3
        relation_status = None
        memory_points = None
        group_nick_name = None

    class DummyPerson:
        def __init__(self) -> None:
            self.person_id = "pid-1"
            self.user_id = ""
            self.platform = ""
            self.is_known = False
            self.nickname = ""
            self.person_name = None
            self.name_reason = None
            self.know_times = 0
            self.relation_status = None
            self.memory_points = ["placeholder"]
            self.group_nick_name = ["placeholder"]

        def sync_to_database(self) -> None:
            raise AssertionError("record exists; sync_to_database should not be called")

    monkeypatch.setattr(
        "src.person_info.person_persistence.PersonInfo.get_or_none",
        lambda *args, **kwargs: FakeRecord(),
    )

    person = DummyPerson()
    load_person_from_database(person)

    assert person.memory_points == []
    assert person.group_nick_name == []
    assert person.person_name == "名字"
    assert person.nickname == "昵称"
