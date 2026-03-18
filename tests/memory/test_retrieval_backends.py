from __future__ import annotations

import pytest

import src.memory_system.retrieval_backends as retrieval_backends


@pytest.mark.asyncio
async def test_fetch_person_memory_uses_resolved_person_filter(monkeypatch):
    monkeypatch.setattr(
        retrieval_backends, "get_person_id_by_person_name", lambda name: "person-1" if name == "张三" else None
    )
    monkeypatch.setattr(retrieval_backends, "get_person_id_by_alias", lambda name: None)

    async def fake_search_memory(query, k, filter_fn):
        assert query == "喜欢什么"
        assert k == 5
        assert filter_fn({"person_id": "person-1"}) is True
        assert filter_fn({"person_id": "person-2"}) is False
        return [
            {"text": "喜欢咖啡", "similarity": 0.91, "category": "偏好", "person_id": "person-1"},
        ]

    monkeypatch.setattr(retrieval_backends.vector_manager, "search_memory", fake_search_memory)

    payload = await retrieval_backends.fetch_person_memory("喜欢什么", person_name="张三")

    assert payload["success"] is True
    assert payload["data"]["results"][0]["content"] == "喜欢咖啡"


@pytest.mark.asyncio
async def test_fetch_lpmm_knowledge_returns_success_and_failure(monkeypatch):
    monkeypatch.setattr(retrieval_backends.global_config.lpmm_knowledge, "enable", True)

    class FakeQaManager:
        async def get_knowledge(self, query, limit=5):
            assert query == "测试问题"
            assert limit == 2
            return "知识命中"

    monkeypatch.setattr("src.chat.knowledge.get_qa_manager", lambda: FakeQaManager())
    payload = await retrieval_backends.fetch_lpmm_knowledge("测试问题", limit=2)

    assert payload["success"] is True
    assert payload["message"] == "你从LPMM知识库中找到以下信息"
    assert payload["data"]["result"] == "知识命中"

    class EmptyQaManager:
        async def get_knowledge(self, query, limit=5):
            del query, limit
            return ""

    monkeypatch.setattr("src.chat.knowledge.get_qa_manager", lambda: EmptyQaManager())
    miss_payload = await retrieval_backends.fetch_lpmm_knowledge("测试问题", limit=2)

    assert miss_payload["success"] is False
