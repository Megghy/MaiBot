from __future__ import annotations

import pytest

from src.memory_system.memory_writer import MemoryWriteRequest, persist_memory_request


class FakePerson:
    def __init__(self) -> None:
        self.calls = []

    async def async_add_memory_point(self, memory_content: str, **kwargs) -> bool:
        self.calls.append(("memory", memory_content, kwargs))
        return True

    async def async_add_group_memory_point(self, group_id: str, memory_content: str, **kwargs) -> bool:
        self.calls.append(("group", group_id, memory_content, kwargs))
        return True


@pytest.mark.asyncio
async def test_persist_memory_request_uses_async_global_memory_path():
    person = FakePerson()
    request = MemoryWriteRequest(
        person=person,  # type: ignore[arg-type]
        memory_content="喜欢吃拉面",
        category="喜好",
        weight=1.5,
        source="unit-test",
        context="测试上下文",
    )

    persisted = await persist_memory_request(request)

    assert persisted is True
    assert person.calls == [
        (
            "memory",
            "喜欢吃拉面",
            {
                "category": "喜好",
                "weight": 1.5,
                "context": "测试上下文",
                "source": "unit-test",
            },
        )
    ]


@pytest.mark.asyncio
async def test_persist_memory_request_uses_async_group_memory_path():
    person = FakePerson()
    request = MemoryWriteRequest(
        person=person,  # type: ignore[arg-type]
        memory_content="在这个群里经常聊游戏",
        category="经历",
        weight=1.0,
        group_id="123456",
    )

    persisted = await persist_memory_request(request)

    assert persisted is True
    assert person.calls == [
        (
            "group",
            "123456",
            "在这个群里经常聊游戏",
            {
                "category": "经历",
                "weight": 1.0,
                "source": None,
                "context": None,
            },
        )
    ]
