from __future__ import annotations

from types import SimpleNamespace

import pytest

import src.memory_system.retrieval_agent as retrieval_agent
from src.llm_models.payload_content.tool_option import ToolCall


def test_resolve_terminal_tool_calls_returns_found_answer():
    step = {"iteration": 1, "thought": "x", "actions": [], "observations": []}
    thinking_steps = []

    outcome = retrieval_agent._resolve_terminal_tool_calls(
        [ToolCall("call-1", "found_answer", {"answer": "最终答案", "source": "memory"})],
        step,
        thinking_steps,
        iteration=0,
    )

    assert outcome == retrieval_agent.TerminalToolOutcome(found_answer=True, payload="最终答案")
    assert thinking_steps == [
        {
            "iteration": 1,
            "thought": "x",
            "actions": [{"action_type": "found_answer", "action_params": {"answer": "最终答案", "source": "memory"}}],
            "observations": ["Agent调用found_answer提交答案"],
        }
    ]


@pytest.mark.asyncio
async def test_execute_tool_calls_injects_chat_id_and_preserves_order():
    class FakeTool:
        def __init__(self, execute_func):
            self.execute_func = execute_func

        async def execute(self, **kwargs):
            return await self.execute_func(**kwargs)

    async def fake_execute(query: str, chat_id: str):
        assert query == "喜欢什么"
        assert chat_id == "chat-1"
        return "命中结果"

    registry = SimpleNamespace(get_tool=lambda name: FakeTool(fake_execute) if name == "query_person_memory" else None)

    outcomes = await retrieval_agent._execute_tool_calls(
        registry,
        [
            ToolCall("call-1", "query_person_memory", {"query": "喜欢什么"}),
            ToolCall("call-2", "missing_tool", {}),
        ],
        chat_id="chat-1",
        iteration=0,
        is_final_iteration=False,
    )

    assert outcomes[0].action_type == "query_person_memory"
    assert outcomes[0].tool_call_id == "call-1"
    assert "查询query_person_memory(query=喜欢什么)的结果：命中结果" == outcomes[0].observation
    assert outcomes[1].observation == "查询missing_tool失败: 未知的工具类型: missing_tool"


@pytest.mark.asyncio
async def test_execute_tool_calls_ignores_non_terminal_tools_in_final_iteration():
    registry = SimpleNamespace(get_tool=lambda name: None)

    outcomes = await retrieval_agent._execute_tool_calls(
        registry,
        [ToolCall("call-1", "query_jargon", {"query": "yyds"})],
        chat_id="chat-1",
        iteration=1,
        is_final_iteration=True,
    )

    assert outcomes == [
        retrieval_agent.ToolExecutionOutcome(
            action_type="query_jargon",
            action_params={"query": "yyds"},
            observation="最终迭代忽略检索工具调用: query_jargon",
            tool_call_id=None,
        )
    ]


@pytest.mark.asyncio
async def test_react_agent_solve_question_returns_terminal_answer(monkeypatch):
    async def fake_format_prompt(name: str, **kwargs):
        assert name == "memory_retrieval_react_prompt_head"
        assert kwargs["question"] == "问题"
        return "prompt"

    class FakeLLM:
        async def generate_response_with_message_async(self, *, message_factory, tools=None):
            messages = message_factory(None)
            assert messages[0].content == "prompt"
            assert any(item["name"] == "found_answer" for item in (tools or []))
            return (
                "我已经找到答案",
                (
                    "",
                    "fake-model",
                    [ToolCall("call-1", "found_answer", {"answer": "答案", "source": "memory"})],
                ),
            )

    monkeypatch.setattr(retrieval_agent.global_prompt_manager, "format_prompt", fake_format_prompt)
    monkeypatch.setattr(retrieval_agent.AgentRunner, "_build_llm_request", lambda self, profile: FakeLLM())
    monkeypatch.setattr(retrieval_agent, "_build_lpmm_hint", lambda: "")
    monkeypatch.setattr(retrieval_agent, "_calculate_iteration_budget", lambda question, collected_info: 2)

    found, answer, thinking_steps, is_timeout = await retrieval_agent._react_agent_solve_question(
        "问题",
        "chat-1",
        max_iterations=2,
        timeout=30.0,
    )

    assert found is True
    assert answer == "答案"
    assert is_timeout is False
    assert thinking_steps[0]["actions"][0]["action_type"] == "found_answer"
