from __future__ import annotations

import pytest

import src.memory_system.memory_ingester as memory_ingester
from src.memory_system.retrieval_analysis import QAAnalysisEntry


@pytest.mark.asyncio
async def test_ingest_question_answer_routes_structured_entries(monkeypatch):
    stored_person_entries = []
    stored_jargon_entries = []

    async def fake_analyze_question_answer(question: str, answer: str):
        assert question == "她喜欢什么？"
        assert answer == "她喜欢拉面，也常说 yyds。"
        return [
            QAAnalysisEntry(category="人物信息", confidence=0.92, person_name="小李", memory_content="喜欢拉面"),
            QAAnalysisEntry(category="黑话", confidence=0.88, jargon_keyword="yyds"),
        ]

    async def fake_store_person_memory_from_answer(person_name: str, memory_content: str, chat_id: str):
        stored_person_entries.append((person_name, memory_content, chat_id))

    async def fake_store_jargon_from_answer(jargon_keyword: str, answer: str, chat_id: str):
        stored_jargon_entries.append((jargon_keyword, answer, chat_id))

    monkeypatch.setattr(memory_ingester, "analyze_question_answer", fake_analyze_question_answer)
    monkeypatch.setattr(memory_ingester, "store_person_memory_from_answer", fake_store_person_memory_from_answer)
    monkeypatch.setattr(memory_ingester, "store_jargon_from_answer", fake_store_jargon_from_answer)

    await memory_ingester.ingest_question_answer("她喜欢什么？", "她喜欢拉面，也常说 yyds。", "chat-1")

    assert stored_person_entries == [("小李", "喜欢拉面", "chat-1")]
    assert stored_jargon_entries == [("yyds", "她喜欢拉面，也常说 yyds。", "chat-1")]


@pytest.mark.asyncio
async def test_ingest_jargon_entries_normalizes_and_filters_blank_values(monkeypatch):
    stored_entries = []

    async def fake_store_jargon_from_answer(jargon_keyword: str, answer: str, chat_id: str):
        stored_entries.append((jargon_keyword, answer, chat_id))

    monkeypatch.setattr(memory_ingester, "store_jargon_from_answer", fake_store_jargon_from_answer)

    await memory_ingester.ingest_jargon_entries(
        {
            " 术语A ": " 含义A ",
            "": "无效",
            "术语B": "   ",
        },
        "chat-2",
    )

    assert stored_entries == [("术语A", "含义A", "chat-2")]
