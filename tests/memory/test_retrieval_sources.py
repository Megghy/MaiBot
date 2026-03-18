from __future__ import annotations

import pytest

import src.memory_system.retrieval_sources as retrieval_sources
from src.memory_system.retrieval_sources import (
    CACHED_MEMORIES_SOURCE,
    CONCEPT_EXPANSION_SOURCE,
    LPMM_KNOWLEDGE_SOURCE,
    RetrievalSourceRequest,
    RetrievalSourceResult,
    collect_retrieval_source_results,
)


class FakeSource:
    def __init__(self, name: str, content: str):
        self.name = name
        self.content = content

    async def collect(self, request: RetrievalSourceRequest) -> RetrievalSourceResult:
        assert request.chat_id == "chat-1"
        return RetrievalSourceResult(source_name=self.name, content=self.content)


class BrokenSource:
    name = "broken"

    async def collect(self, request: RetrievalSourceRequest) -> RetrievalSourceResult:
        del request
        raise RuntimeError("boom")


@pytest.mark.asyncio
async def test_collect_retrieval_source_results_returns_named_results():
    request = RetrievalSourceRequest(chat_id="chat-1")

    results = await collect_retrieval_source_results(
        request,
        [FakeSource("one", "alpha"), FakeSource("two", "beta")],
    )

    assert list(results) == ["one", "two"]
    assert results["one"].content == "alpha"
    assert results["two"].content == "beta"


@pytest.mark.asyncio
async def test_collect_retrieval_source_results_isolates_source_errors():
    request = RetrievalSourceRequest(chat_id="chat-1")

    results = await collect_retrieval_source_results(
        request,
        [BrokenSource(), FakeSource("ok", "value")],
    )

    assert results["broken"].content == ""
    assert results["ok"].content == "value"


@pytest.mark.asyncio
async def test_cached_memories_source_returns_payload_and_joined_content(monkeypatch):
    monkeypatch.setattr(retrieval_sources, "_get_cached_memories", lambda *args, **kwargs: ["a", "b"])

    result = await CACHED_MEMORIES_SOURCE.collect(RetrievalSourceRequest(chat_id="chat-1"))

    assert result.payload == ["a", "b"]
    assert result.content == "a\n\nb"
    assert result.metadata["count"] == 2


@pytest.mark.asyncio
async def test_concept_and_lpmm_sources_filter_results(monkeypatch):
    async def fake_retrieve_concepts(concepts, chat_id, context=""):
        assert concepts == ["术语A"]
        assert chat_id == "chat-1"
        assert context == "focus"
        return "concept-hit"

    async def fake_query_lpmm_hit(question: str, limit: int = 5):
        assert question == "问题"
        assert limit == 2
        return "你从LPMM知识库中找到以下信息"

    monkeypatch.setattr(retrieval_sources, "_retrieve_concepts_with_jargon", fake_retrieve_concepts)
    monkeypatch.setattr(retrieval_sources, "query_lpmm_knowledge", fake_query_lpmm_hit)

    concept_result = await CONCEPT_EXPANSION_SOURCE.collect(
        RetrievalSourceRequest(chat_id="chat-1", concepts=("术语A",), focus_summary="focus")
    )
    lpmm_result = await LPMM_KNOWLEDGE_SOURCE.collect(
        RetrievalSourceRequest(chat_id="chat-1", question="问题", result_limit=2)
    )

    assert concept_result.content == "concept-hit"
    assert concept_result.metadata["concepts"] == ["术语A"]
    assert lpmm_result.content == "你从LPMM知识库中找到以下信息"

    async def fake_query_lpmm_miss(question: str, limit: int = 5):
        del question, limit
        return "未找到相关信息"

    monkeypatch.setattr(retrieval_sources, "query_lpmm_knowledge", fake_query_lpmm_miss)
    lpmm_miss_result = await LPMM_KNOWLEDGE_SOURCE.collect(
        RetrievalSourceRequest(chat_id="chat-1", question="问题", result_limit=2)
    )

    assert lpmm_miss_result.content == ""
