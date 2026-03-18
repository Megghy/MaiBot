from __future__ import annotations

import asyncio
from collections.abc import Mapping

from src.common.logger import get_logger
from src.jargon.jargon_miner import store_jargon_from_answer
from src.memory_system.retrieval_analysis import analyze_question_answer
from src.person_info.person_info import store_person_memory_from_answer

logger = get_logger("memory_ingester")


async def ingest_question_answer(question: str, answer: str, chat_id: str) -> None:
    try:
        entries = await analyze_question_answer(question, answer)
        if not entries:
            logger.info(f"本次 QA 分析未提取可存储的黑话/人物信息，问题: {question[:50]}...")
            return

        for entry in entries:
            if entry.category == "黑话":
                await store_jargon_from_answer(entry.jargon_keyword, answer, chat_id)
                continue
            if entry.category == "人物信息":
                await store_person_memory_from_answer(entry.person_name, entry.memory_content, chat_id)
    except Exception as exc:
        logger.error(f"写入 QA 衍生记忆失败: {exc}")


async def ingest_jargon_entries(jargon_entries: Mapping[str, str], chat_id: str) -> None:
    try:
        normalized_entries = {
            str(term).strip()[:64]: str(meaning).strip()[:200]
            for term, meaning in jargon_entries.items()
            if str(term).strip() and str(meaning).strip()
        }
        if not normalized_entries:
            return

        for term, meaning in normalized_entries.items():
            await store_jargon_from_answer(term, meaning, chat_id)
    except Exception as exc:
        logger.error(f"写入概念回退黑话失败: {exc}")


def schedule_question_answer_ingestion(question: str, answer: str, chat_id: str) -> None:
    asyncio.create_task(
        ingest_question_answer(question, answer, chat_id),
        name="memory_ingester.question_answer",
    )


def schedule_jargon_ingestion(jargon_entries: Mapping[str, str], chat_id: str) -> None:
    if not jargon_entries:
        return
    asyncio.create_task(
        ingest_jargon_entries(dict(jargon_entries), chat_id),
        name="memory_ingester.jargon",
    )


__all__ = [
    "ingest_jargon_entries",
    "ingest_question_answer",
    "schedule_jargon_ingestion",
    "schedule_question_answer_ingestion",
]
