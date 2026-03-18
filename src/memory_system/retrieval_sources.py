from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from src.common.logger import get_logger
from src.config.config import global_config
from src.memory_system.retrieval_backends import (
    fetch_chat_history,
    fetch_jargon,
    fetch_lpmm_knowledge,
    fetch_person_info,
    fetch_person_memory,
)
from src.memory_system.retrieval_cache import _get_cached_memory_entries, _get_recent_query_history
from src.memory_system.retrieval_concepts import _retrieve_concepts_with_jargon

logger = get_logger("memory_retrieval")


@dataclass(slots=True)
class RetrievalSourceRequest:
    chat_id: str
    question: str = ""
    context: str = ""
    focus_summary: str = ""
    concepts: tuple[str, ...] = ()
    keyword: str = ""
    time_range: str = ""
    person_name: str = ""
    fuzzy: bool = True
    time_window_seconds: float = 300.0
    result_limit: int = 5


@dataclass(slots=True)
class RetrievalSourceResult:
    source_name: str
    content: str = ""
    payload: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_content(self) -> bool:
        return bool(self.content.strip()) or self.payload not in (None, [], (), {})


class RetrievalSource(Protocol):
    name: str

    async def collect(self, request: RetrievalSourceRequest) -> RetrievalSourceResult:
        raise NotImplementedError


def _result_from_backend_payload(
    source_name: str,
    payload: dict[str, Any],
    *,
    metadata: dict[str, Any] | None = None,
) -> RetrievalSourceResult:
    return RetrievalSourceResult(
        source_name=source_name,
        content=str(payload.get("message") or ""),
        payload=payload,
        metadata=metadata or {},
    )


class RecentQueryHistorySource:
    name = "recent_query_history"

    async def collect(self, request: RetrievalSourceRequest) -> RetrievalSourceResult:
        history = await asyncio.to_thread(
            _get_recent_query_history,
            request.chat_id,
            time_window_seconds=request.time_window_seconds,
        )
        return RetrievalSourceResult(
            source_name=self.name,
            content=history or "",
        )


class CachedMemoriesSource:
    name = "cached_memories"

    async def collect(self, request: RetrievalSourceRequest) -> RetrievalSourceResult:
        memory_entries = await asyncio.to_thread(
            _get_cached_memory_entries,
            request.chat_id,
            time_window_seconds=request.time_window_seconds,
        )
        return RetrievalSourceResult(
            source_name=self.name,
            content="\n\n".join(entry.block for entry in memory_entries),
            payload=list(memory_entries),
            metadata={"count": len(memory_entries)},
        )


class ConceptExpansionSource:
    name = "concept_expansion"

    async def collect(self, request: RetrievalSourceRequest) -> RetrievalSourceResult:
        concepts = tuple(concept.strip() for concept in request.concepts if concept and concept.strip())
        if not concepts:
            return RetrievalSourceResult(source_name=self.name)

        retrieval_result = await _retrieve_concepts_with_jargon(
            list(concepts),
            request.chat_id,
            context=request.focus_summary,
        )
        return RetrievalSourceResult(
            source_name=self.name,
            content=retrieval_result.content,
            payload=retrieval_result.learned_jargon,
            metadata={
                "concepts": list(concepts),
                "learned_jargon_count": len(retrieval_result.learned_jargon),
            },
        )


class LpmmKnowledgeSource:
    name = "lpmm_knowledge"

    async def collect(self, request: RetrievalSourceRequest) -> RetrievalSourceResult:
        question = request.question.strip()
        if not question or not global_config.lpmm_knowledge.enable:
            return RetrievalSourceResult(source_name=self.name)

        payload = await fetch_lpmm_knowledge(question, limit=request.result_limit)
        if not payload["success"]:
            content = ""
        else:
            content = payload["message"]
        if not content.startswith("你从LPMM知识库中找到"):
            content = ""
        return RetrievalSourceResult(
            source_name=self.name,
            content=content,
            payload=payload,
            metadata={"question": question, "limit": request.result_limit},
        )


class ChatHistorySource:
    name = "chat_history"

    async def collect(self, request: RetrievalSourceRequest) -> RetrievalSourceResult:
        keyword = request.keyword.strip() or None
        time_range = request.time_range.strip() or None
        payload = await fetch_chat_history(
            request.chat_id,
            keyword=keyword,
            time_range=time_range,
            fuzzy=request.fuzzy,
            result_limit=request.result_limit,
        )
        return _result_from_backend_payload(
            self.name,
            payload,
            metadata={
                "chat_id": request.chat_id,
                "keyword": keyword,
                "time_range": time_range,
                "fuzzy": request.fuzzy,
                "limit": request.result_limit,
            },
        )


class PersonInfoSource:
    name = "person_info"

    async def collect(self, request: RetrievalSourceRequest) -> RetrievalSourceResult:
        person_name = request.person_name.strip() or request.question.strip()
        if not person_name:
            return RetrievalSourceResult(source_name=self.name)

        payload = await fetch_person_info(person_name, result_limit=request.result_limit)
        return _result_from_backend_payload(
            self.name,
            payload,
            metadata={"person_name": person_name, "limit": request.result_limit},
        )


class PersonMemorySource:
    name = "person_memory"

    async def collect(self, request: RetrievalSourceRequest) -> RetrievalSourceResult:
        query = request.question.strip()
        if not query:
            return RetrievalSourceResult(source_name=self.name)

        payload = await fetch_person_memory(
            query,
            person_name=request.person_name.strip() or None,
            result_limit=request.result_limit,
        )
        return _result_from_backend_payload(
            self.name,
            payload,
            metadata={"query": query, "person_name": request.person_name.strip(), "limit": request.result_limit},
        )


class JargonSource:
    name = "jargon"

    async def collect(self, request: RetrievalSourceRequest) -> RetrievalSourceResult:
        query = request.question.strip()
        if not query:
            return RetrievalSourceResult(source_name=self.name)

        payload = await fetch_jargon(query, result_limit=request.result_limit)
        return _result_from_backend_payload(
            self.name,
            payload,
            metadata={"query": query, "limit": request.result_limit},
        )


async def collect_retrieval_source_results(
    request: RetrievalSourceRequest,
    sources: Sequence[RetrievalSource],
) -> dict[str, RetrievalSourceResult]:
    results: dict[str, RetrievalSourceResult] = {}

    for source in sources:
        try:
            result = await source.collect(request)
        except Exception as exc:
            logger.error(f"检索源 {source.name} 执行失败: {exc}")
            result = RetrievalSourceResult(source_name=source.name)
        results[source.name] = result

    return results


RECENT_QUERY_HISTORY_SOURCE = RecentQueryHistorySource()
CACHED_MEMORIES_SOURCE = CachedMemoriesSource()
CONCEPT_EXPANSION_SOURCE = ConceptExpansionSource()
LPMM_KNOWLEDGE_SOURCE = LpmmKnowledgeSource()
CHAT_HISTORY_SOURCE = ChatHistorySource()
PERSON_INFO_SOURCE = PersonInfoSource()
PERSON_MEMORY_SOURCE = PersonMemorySource()
JARGON_SOURCE = JargonSource()


__all__ = [
    "CACHED_MEMORIES_SOURCE",
    "CHAT_HISTORY_SOURCE",
    "CONCEPT_EXPANSION_SOURCE",
    "JARGON_SOURCE",
    "LPMM_KNOWLEDGE_SOURCE",
    "PERSON_INFO_SOURCE",
    "PERSON_MEMORY_SOURCE",
    "RECENT_QUERY_HISTORY_SOURCE",
    "RetrievalSource",
    "RetrievalSourceRequest",
    "RetrievalSourceResult",
    "collect_retrieval_source_results",
]
