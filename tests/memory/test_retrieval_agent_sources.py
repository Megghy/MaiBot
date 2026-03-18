from __future__ import annotations

import pytest

import src.memory_system.retrieval_sources as retrieval_sources
from src.memory_system.retrieval_sources import (
    CHAT_HISTORY_SOURCE,
    JARGON_SOURCE,
    LPMM_KNOWLEDGE_SOURCE,
    PERSON_INFO_SOURCE,
    PERSON_MEMORY_SOURCE,
    RetrievalSourceRequest,
)


@pytest.mark.asyncio
async def test_chat_history_source_delegates_to_backend(monkeypatch):
    async def fake_fetch_chat_history(chat_id, **kwargs):
        assert chat_id == "chat-1"
        assert kwargs["keyword"] == "GitHub"
        assert kwargs["time_range"] == "2026-03-18 10:00:00"
        assert kwargs["fuzzy"] is False
        assert kwargs["result_limit"] == 2
        return {"success": True, "message": "chat", "data": {"records": []}}

    monkeypatch.setattr(retrieval_sources, "fetch_chat_history", fake_fetch_chat_history)

    result = await CHAT_HISTORY_SOURCE.collect(
        RetrievalSourceRequest(
            chat_id="chat-1",
            keyword="GitHub",
            time_range="2026-03-18 10:00:00",
            fuzzy=False,
            result_limit=2,
        )
    )

    assert result.content == "chat"
    assert result.payload["success"] is True
    assert result.metadata["keyword"] == "GitHub"


@pytest.mark.asyncio
async def test_person_and_vector_sources_delegate_to_backends(monkeypatch):
    async def fake_fetch_person_info(person_name, **kwargs):
        assert person_name == "张三"
        assert kwargs["result_limit"] == 4
        return {"success": True, "message": "person", "data": {"records": []}}

    async def fake_fetch_person_memory(query, **kwargs):
        assert query == "喜欢什么"
        assert kwargs["person_name"] == "张三"
        assert kwargs["result_limit"] == 3
        return {"success": True, "message": "memory", "data": {"results": []}}

    async def fake_fetch_jargon(query, **kwargs):
        assert query == "yyds"
        assert kwargs["result_limit"] == 5
        return {"success": True, "message": "jargon", "data": {"results": []}}

    monkeypatch.setattr(retrieval_sources, "fetch_person_info", fake_fetch_person_info)
    monkeypatch.setattr(retrieval_sources, "fetch_person_memory", fake_fetch_person_memory)
    monkeypatch.setattr(retrieval_sources, "fetch_jargon", fake_fetch_jargon)

    person_result = await PERSON_INFO_SOURCE.collect(
        RetrievalSourceRequest(chat_id="", person_name="张三", result_limit=4)
    )
    memory_result = await PERSON_MEMORY_SOURCE.collect(
        RetrievalSourceRequest(chat_id="chat-1", question="喜欢什么", person_name="张三", result_limit=3)
    )
    jargon_result = await JARGON_SOURCE.collect(
        RetrievalSourceRequest(chat_id="chat-1", question="yyds", result_limit=5)
    )

    assert person_result.payload["message"] == "person"
    assert memory_result.metadata["person_name"] == "张三"
    assert jargon_result.content == "jargon"


@pytest.mark.asyncio
async def test_lpmm_source_preserves_backend_payload(monkeypatch):
    async def fake_fetch_lpmm_knowledge(query, **kwargs):
        assert query == "知识点"
        assert kwargs["limit"] == 2
        return {"success": True, "message": "你从LPMM知识库中找到以下信息", "data": {"result": "命中"}}

    monkeypatch.setattr(retrieval_sources, "fetch_lpmm_knowledge", fake_fetch_lpmm_knowledge)

    result = await LPMM_KNOWLEDGE_SOURCE.collect(RetrievalSourceRequest(chat_id="", question="知识点", result_limit=2))

    assert result.content == "你从LPMM知识库中找到以下信息"
    assert result.payload["data"]["result"] == "命中"
    assert result.metadata["limit"] == 2
