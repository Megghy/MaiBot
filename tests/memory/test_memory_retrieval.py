from __future__ import annotations

from types import SimpleNamespace

import pytest

import src.memory_system.memory_retrieval as memory_retrieval
from src.memory_system.memory_retrieval import QuestionRetrievalResult
from src.memory_system.retrieval_cache import CachedMemoryEntry


@pytest.mark.asyncio
async def test_process_single_question_returns_structured_result_for_cache_hit(monkeypatch):
    thinking_steps = [
        {
            "actions": [
                {
                    "action_type": "found_answer",
                    "action_params": {"source": "memory"},
                }
            ]
        }
    ]

    async def fake_collect_retrieval_source_results(request, sources):
        assert request.chat_id == "chat-1"
        assert request.question == "问题1"
        assert [source.name for source in sources] == [memory_retrieval.LPMM_KNOWLEDGE_SOURCE.name]
        return {
            memory_retrieval.LPMM_KNOWLEDGE_SOURCE.name: SimpleNamespace(content="", payload=None),
        }

    async def fail_react_agent(*args, **kwargs):
        raise AssertionError("cache hit should not invoke react agent")

    monkeypatch.setattr(memory_retrieval, "collect_retrieval_source_results", fake_collect_retrieval_source_results)
    monkeypatch.setattr(memory_retrieval, "_cleanup_stale_not_found_thinking_back", lambda: None)
    monkeypatch.setattr(
        memory_retrieval,
        "_query_thinking_back",
        lambda chat_id, question: (1, "缓存答案", thinking_steps),
    )
    monkeypatch.setattr(memory_retrieval.random, "random", lambda: 0.0)
    monkeypatch.setattr(memory_retrieval, "_react_agent_solve_question", fail_react_agent)

    result = await memory_retrieval._process_single_question("问题1", "chat-1", "上下文")

    assert result == QuestionRetrievalResult(
        question="问题1",
        answer="缓存答案",
        thinking_steps=thinking_steps,
        from_cache=True,
        should_ingest=False,
    )


@pytest.mark.asyncio
async def test_build_memory_retrieval_prompt_schedules_ingestion_at_top_boundary(monkeypatch):
    scheduled_qa_ingestions = []
    scheduled_jargon_ingestions = []

    async def fake_collect_retrieval_source_results(request, sources):
        source_names = [source.name for source in sources]
        if source_names == [memory_retrieval.RECENT_QUERY_HISTORY_SOURCE.name]:
            return {
                memory_retrieval.RECENT_QUERY_HISTORY_SOURCE.name: SimpleNamespace(content="最近查询记录"),
            }
        if source_names == [
            memory_retrieval.CONCEPT_EXPANSION_SOURCE.name,
            memory_retrieval.CACHED_MEMORIES_SOURCE.name,
        ]:
            return {
                memory_retrieval.CONCEPT_EXPANSION_SOURCE.name: SimpleNamespace(
                    content="【概念检索结果】\n'术语A' 的含义为：解释A\n",
                    payload={"术语A": "解释A"},
                ),
                memory_retrieval.CACHED_MEMORIES_SOURCE.name: SimpleNamespace(content="", payload=[]),
            }
        raise AssertionError(source_names)

    async def fake_format_prompt(name: str, **kwargs):
        assert name == "memory_retrieval_question_prompt"
        assert kwargs["sender"] == "用户"
        return "prompt"

    async def fake_generate_with_model(prompt: str, **kwargs):
        assert prompt == "prompt"
        assert kwargs["request_type"] == "memory.question"
        return True, '{"concepts":["术语A"],"questions":["问题1"]}', "", "model"

    async def fake_record_question_generation_trace(*args, **kwargs):
        return None

    async def fake_process_single_question(**kwargs):
        assert kwargs["question"] == "问题1"
        return QuestionRetrievalResult(
            question="问题1",
            answer="答案1",
            thinking_steps=[],
            should_ingest=True,
        )

    monkeypatch.setattr(memory_retrieval, "collect_retrieval_source_results", fake_collect_retrieval_source_results)
    monkeypatch.setattr(memory_retrieval.global_prompt_manager, "format_prompt", fake_format_prompt)
    monkeypatch.setattr(memory_retrieval.llm_api, "generate_with_model", fake_generate_with_model)
    monkeypatch.setattr(memory_retrieval, "_build_participant_hints", lambda chat_id: f"hints:{chat_id}")
    monkeypatch.setattr(memory_retrieval, "_extract_keywords", lambda target: ["术语A"])
    monkeypatch.setattr(memory_retrieval, "_build_focus_summary", lambda *args, **kwargs: "focus")
    monkeypatch.setattr(memory_retrieval, "_parse_questions_json", lambda response: (["术语A"], ["问题1"]))
    monkeypatch.setattr(memory_retrieval, "_record_question_generation_trace", fake_record_question_generation_trace)
    monkeypatch.setattr(memory_retrieval, "_process_single_question", fake_process_single_question)
    monkeypatch.setattr(
        memory_retrieval,
        "schedule_question_answer_ingestion",
        lambda question, answer, chat_id: scheduled_qa_ingestions.append((question, answer, chat_id)),
    )
    monkeypatch.setattr(
        memory_retrieval,
        "schedule_jargon_ingestion",
        lambda jargon_entries, chat_id: scheduled_jargon_ingestions.append((dict(jargon_entries), chat_id)),
    )
    monkeypatch.setattr(
        memory_retrieval,
        "_format_answer_payload",
        lambda question, answer, thinking_steps=None, from_cache=False, extra_note=None: (
            f"问题：{question}\n答案：{answer}"
        ),
    )

    prompt = await memory_retrieval.build_memory_retrieval_prompt(
        message="聊天记录",
        sender="用户",
        target="术语A 是什么意思？",
        chat_stream=SimpleNamespace(stream_id="chat-1"),
        tool_executor=None,
    )

    assert scheduled_jargon_ingestions == [({"术语A": "解释A"}, "chat-1")]
    assert scheduled_qa_ingestions == [("问题1", "答案1", "chat-1")]
    assert "你回忆起了以下信息" in prompt
    assert "问题：问题1" in prompt


@pytest.mark.asyncio
async def test_build_memory_retrieval_prompt_merges_and_ranks_question_hits_and_cached_memories(monkeypatch):
    async def fake_collect_retrieval_source_results(request, sources):
        source_names = [source.name for source in sources]
        if source_names == [memory_retrieval.RECENT_QUERY_HISTORY_SOURCE.name]:
            return {
                memory_retrieval.RECENT_QUERY_HISTORY_SOURCE.name: SimpleNamespace(content="最近查询记录"),
            }
        if source_names == [
            memory_retrieval.CONCEPT_EXPANSION_SOURCE.name,
            memory_retrieval.CACHED_MEMORIES_SOURCE.name,
        ]:
            return {
                memory_retrieval.CONCEPT_EXPANSION_SOURCE.name: SimpleNamespace(
                    content="【概念检索结果】\n'梗A' 的含义为：解释A\n",
                    payload={},
                ),
                memory_retrieval.CACHED_MEMORIES_SOURCE.name: SimpleNamespace(
                    content="",
                    payload=[
                        CachedMemoryEntry(
                            question="问题1",
                            answer="旧缓存答案",
                            thinking_steps=[],
                            block="问题：问题1\n答案：旧缓存答案\n来源：历史缓存",
                        ),
                        CachedMemoryEntry(
                            question="问题3",
                            answer="缓存补充答案",
                            thinking_steps=[],
                            block="问题：问题3\n答案：缓存补充答案\n来源：历史缓存",
                        ),
                    ],
                ),
            }
        raise AssertionError(source_names)

    async def fake_format_prompt(name: str, **kwargs):
        assert name == "memory_retrieval_question_prompt"
        return "prompt"

    async def fake_generate_with_model(prompt: str, **kwargs):
        assert prompt == "prompt"
        return True, '{"concepts":["梗A"],"questions":["问题1","问题2"]}', "", "model"

    async def fake_record_question_generation_trace(*args, **kwargs):
        return None

    async def fake_process_single_question(**kwargs):
        if kwargs["question"] == "问题1":
            return QuestionRetrievalResult(
                question="问题1",
                answer="实时检索答案",
                thinking_steps=[{"actions": [{"action_type": "found_answer", "action_params": {"source": "memory"}}]}],
                should_ingest=False,
            )
        return QuestionRetrievalResult(
            question="问题2",
            answer="模型答案",
            thinking_steps=[{"actions": [{"action_type": "found_answer", "action_params": {"source": "model"}}]}],
            should_ingest=False,
        )

    monkeypatch.setattr(memory_retrieval, "collect_retrieval_source_results", fake_collect_retrieval_source_results)
    monkeypatch.setattr(memory_retrieval.global_prompt_manager, "format_prompt", fake_format_prompt)
    monkeypatch.setattr(memory_retrieval.llm_api, "generate_with_model", fake_generate_with_model)
    monkeypatch.setattr(memory_retrieval, "_build_participant_hints", lambda chat_id: f"hints:{chat_id}")
    monkeypatch.setattr(memory_retrieval, "_extract_keywords", lambda target: ["梗A"])
    monkeypatch.setattr(memory_retrieval, "_build_focus_summary", lambda *args, **kwargs: "focus")
    monkeypatch.setattr(memory_retrieval, "_parse_questions_json", lambda response: (["梗A"], ["问题1", "问题2"]))
    monkeypatch.setattr(memory_retrieval, "_record_question_generation_trace", fake_record_question_generation_trace)
    monkeypatch.setattr(memory_retrieval, "_process_single_question", fake_process_single_question)
    monkeypatch.setattr(memory_retrieval, "schedule_question_answer_ingestion", lambda *args, **kwargs: None)
    monkeypatch.setattr(memory_retrieval, "schedule_jargon_ingestion", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        memory_retrieval,
        "_format_answer_payload",
        lambda question, answer, thinking_steps=None, from_cache=False, extra_note=None: (
            f"问题：{question}\n答案：{answer}\n来源：{'历史缓存' if from_cache else ('检索记忆' if question == '问题1' else '模型常识')}"
        ),
    )

    prompt = await memory_retrieval.build_memory_retrieval_prompt(
        message="聊天记录",
        sender="用户",
        target="帮我回忆一下",
        chat_stream=SimpleNamespace(stream_id="chat-1"),
        tool_executor=None,
    )

    assert prompt.index("【概念检索结果】") < prompt.index("问题：问题1")
    assert prompt.index("问题：问题1") < prompt.index("问题：问题2")
    assert prompt.index("问题：问题2") < prompt.index("问题：问题3")
    assert "旧缓存答案" not in prompt
