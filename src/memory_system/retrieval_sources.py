from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from src.common.logger import get_logger
from src.config.config import global_config
from src.memory_system.retrieval_cache import _get_cached_memories, _get_recent_query_history
from src.memory_system.retrieval_concepts import _retrieve_concepts_with_jargon
from src.memory_system.retrieval_tools.query_lpmm_knowledge import query_lpmm_knowledge

logger = get_logger("memory_retrieval")


@dataclass(slots=True)
class RetrievalSourceRequest:
    chat_id: str
    question: str = ""
    context: str = ""
    focus_summary: str = ""
    concepts: tuple[str, ...] = ()
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
        memories = await asyncio.to_thread(
            _get_cached_memories,
            request.chat_id,
            time_window_seconds=request.time_window_seconds,
        )
        return RetrievalSourceResult(
            source_name=self.name,
            content="\n\n".join(memories),
            payload=list(memories),
            metadata={"count": len(memories)},
        )


class ConceptExpansionSource:
    name = "concept_expansion"

    async def collect(self, request: RetrievalSourceRequest) -> RetrievalSourceResult:
        concepts = tuple(concept.strip() for concept in request.concepts if concept and concept.strip())
        if not concepts:
            return RetrievalSourceResult(source_name=self.name)

        content = await _retrieve_concepts_with_jargon(list(concepts), request.chat_id, context=request.focus_summary)
        return RetrievalSourceResult(
            source_name=self.name,
            content=content or "",
            metadata={"concepts": list(concepts)},
        )


class LpmmKnowledgeSource:
    name = "lpmm_knowledge"

    async def collect(self, request: RetrievalSourceRequest) -> RetrievalSourceResult:
        question = request.question.strip()
        if not question or not global_config.lpmm_knowledge.enable:
            return RetrievalSourceResult(source_name=self.name)

        content = await query_lpmm_knowledge(question, limit=request.result_limit)
        if not content.startswith("你从LPMM知识库中找到"):
            content = ""
        return RetrievalSourceResult(
            source_name=self.name,
            content=content,
            metadata={"question": question, "limit": request.result_limit},
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


__all__ = [
    "CACHED_MEMORIES_SOURCE",
    "CONCEPT_EXPANSION_SOURCE",
    "LPMM_KNOWLEDGE_SOURCE",
    "RECENT_QUERY_HISTORY_SOURCE",
    "RetrievalSource",
    "RetrievalSourceRequest",
    "RetrievalSourceResult",
    "collect_retrieval_source_results",
]
