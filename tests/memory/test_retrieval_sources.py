from __future__ import annotations

import pytest

import src.memory_system.retrieval_sources as retrieval_sources
from src.memory_system.retrieval_cache import CachedMemoryEntry
from src.memory_system.retrieval_concepts import ConceptRetrievalResult
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
    monkeypatch.setattr(
        retrieval_sources,
        "_get_cached_memory_entries",
        lambda *args, **kwargs: [
            CachedMemoryEntry(question="问题1", answer="答案1", thinking_steps=[], block="a"),
            CachedMemoryEntry(question="问题2", answer="答案2", thinking_steps=[], block="b"),
        ],
    )

    result = await CACHED_MEMORIES_SOURCE.collect(RetrievalSourceRequest(chat_id="chat-1"))

    assert [entry.block for entry in result.payload] == ["a", "b"]
    assert result.content == "a\n\nb"
    assert result.metadata["count"] == 2


@pytest.mark.asyncio
async def test_concept_and_lpmm_sources_filter_results(monkeypatch):
    async def fake_retrieve_concepts(concepts, chat_id, context=""):
        assert concepts == ["术语A"]
        assert chat_id == "chat-1"
        assert context == "focus"
        return ConceptRetrievalResult(content="concept-hit", learned_jargon={"术语A": "解释"})

    async def fake_fetch_lpmm_hit(question: str, limit: int = 5):
        assert question == "问题"
        assert limit == 2
        return {
            "success": True,
            "message": "你从LPMM知识库中找到以下信息",
            "data": {"result": "命中"},
        }

    monkeypatch.setattr(retrieval_sources, "_retrieve_concepts_with_jargon", fake_retrieve_concepts)
    monkeypatch.setattr(retrieval_sources, "fetch_lpmm_knowledge", fake_fetch_lpmm_hit)

    concept_result = await CONCEPT_EXPANSION_SOURCE.collect(
        RetrievalSourceRequest(chat_id="chat-1", concepts=("术语A",), focus_summary="focus")
    )
    lpmm_result = await LPMM_KNOWLEDGE_SOURCE.collect(
        RetrievalSourceRequest(chat_id="chat-1", question="问题", result_limit=2)
    )

    assert concept_result.content == "concept-hit"
    assert concept_result.payload == {"术语A": "解释"}
    assert concept_result.metadata["concepts"] == ["术语A"]
    assert concept_result.metadata["learned_jargon_count"] == 1
    assert lpmm_result.content == "你从LPMM知识库中找到以下信息"

    async def fake_fetch_lpmm_miss(question: str, limit: int = 5):
        del question, limit
        return {"success": False, "message": "未找到相关信息", "data": {}}

    monkeypatch.setattr(retrieval_sources, "fetch_lpmm_knowledge", fake_fetch_lpmm_miss)
    lpmm_miss_result = await LPMM_KNOWLEDGE_SOURCE.collect(
        RetrievalSourceRequest(chat_id="chat-1", question="问题", result_limit=2)
    )

    assert lpmm_miss_result.content == ""
