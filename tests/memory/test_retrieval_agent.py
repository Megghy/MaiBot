from __future__ import annotations

import pytest

import src.memory_system.retrieval_agent as retrieval_agent
from src.llm_models.payload_content.tool_option import ToolCall


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
