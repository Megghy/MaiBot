from __future__ import annotations

import json

import pytest

import src.memory_system.retrieval_tools.query_chat_history as query_chat_history_module
import src.memory_system.retrieval_tools.query_lpmm_knowledge as query_lpmm_module
import src.memory_system.retrieval_tools.query_person_info as query_person_info_module
import src.memory_system.retrieval_tools.query_vector as query_vector_module
from src.memory_system.retrieval_sources import RetrievalSourceResult


@pytest.mark.asyncio
async def test_query_chat_history_delegates_to_source(monkeypatch):
    async def fake_collect(request):
        assert request.chat_id == "chat-1"
        assert request.keyword == "关键字"
        return RetrievalSourceResult(
            source_name="chat_history",
            payload={"success": True, "message": "ok", "data": {"records": []}},
        )

    monkeypatch.setattr(query_chat_history_module.CHAT_HISTORY_SOURCE, "collect", fake_collect)

    result = await query_chat_history_module.query_chat_history("chat-1", keyword="关键字")

    assert json.loads(result)["message"] == "ok"


@pytest.mark.asyncio
async def test_query_person_info_and_vector_tools_delegate_to_sources(monkeypatch):
    async def fake_person_collect(request):
        assert request.person_name == "张三"
        return RetrievalSourceResult(
            source_name="person_info",
            payload={"success": True, "message": "person", "data": {"records": []}},
        )

    async def fake_memory_collect(request):
        assert request.question == "记忆"
        assert request.person_name == "张三"
        return RetrievalSourceResult(
            source_name="person_memory",
            payload={"success": True, "message": "memory", "data": {"results": []}},
        )

    async def fake_jargon_collect(request):
        assert request.question == "黑话"
        return RetrievalSourceResult(
            source_name="jargon",
            payload={"success": True, "message": "jargon", "data": {"results": []}},
        )

    monkeypatch.setattr(query_person_info_module.PERSON_INFO_SOURCE, "collect", fake_person_collect)
    monkeypatch.setattr(query_vector_module.PERSON_MEMORY_SOURCE, "collect", fake_memory_collect)
    monkeypatch.setattr(query_vector_module.JARGON_SOURCE, "collect", fake_jargon_collect)

    person_result = await query_person_info_module.query_person_info("张三")
    memory_result = await query_vector_module.query_person_memory("记忆", person_name="张三")
    jargon_result = await query_vector_module.query_jargon("黑话")

    assert json.loads(person_result)["message"] == "person"
    assert json.loads(memory_result)["message"] == "memory"
    assert json.loads(jargon_result)["message"] == "jargon"


@pytest.mark.asyncio
async def test_query_lpmm_knowledge_delegates_to_source(monkeypatch):
    async def fake_collect(request):
        assert request.question == "知识"
        assert request.result_limit == 3
        return RetrievalSourceResult(
            source_name="lpmm_knowledge",
            payload={"success": True, "message": "lpmm", "data": {"result": "x"}},
        )

    monkeypatch.setattr(query_lpmm_module.LPMM_KNOWLEDGE_SOURCE, "collect", fake_collect)

    result = await query_lpmm_module.query_lpmm_knowledge("知识", limit=3)

    assert json.loads(result)["message"] == "lpmm"
