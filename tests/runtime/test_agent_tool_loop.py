from __future__ import annotations

import pytest

from src.llm_models.payload_content.tool_option import ToolCall, ToolParamType
from src.runtime.agent import (
    AgentMode,
    AgentProfile,
    AgentRequest,
    AgentRunner,
    AgentStatus,
    PromptBlock,
    PromptBlockRole,
    ToolExecutionContext,
    ToolParamSpec,
    ToolRegistry,
    ToolSpec,
)


@pytest.mark.asyncio
async def test_agent_runner_tool_loop_executes_tool_then_terminal_answer():
    observed_args = []

    async def fake_tool(context: ToolExecutionContext, args: dict):
        observed_args.append((context.chat_id, args))
        return "memory result"

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="query_memory",
            description="query memory",
            params_schema=[ToolParamSpec("query", ToolParamType.STRING, "query", True)],
            executor=fake_tool,
            source="memory",
            tags={"memory"},
        )
    )

    class FakeLLM:
        def __init__(self):
            self.calls = 0

        async def generate_response_with_message_async(self, *, message_factory, tools=None):
            self.calls += 1
            messages = message_factory(None)
            assert messages[0].content == "system"
            assert messages[1].content == "user"
            if self.calls == 1:
                return "", ("", "fake-model", [ToolCall("call-1", "query_memory", {"query": "x"})])
            return "", ("", "fake-model", [ToolCall("call-2", "found_answer", {"answer": "done", "source": "memory"})])

    result = await AgentRunner(tool_registry=registry).run(
        AgentRequest(
            request_id="req-1",
            chat_id="chat-1",
            user_id=None,
            message="question",
            prompt_blocks=[
                PromptBlock(name="policy", content="system", role=PromptBlockRole.SYSTEM),
                PromptBlock(name="task", content="user", role=PromptBlockRole.USER),
            ],
        ),
        AgentProfile(
            name="memory_retrieval",
            skill_name="memory_retrieval",
            agent_mode=AgentMode.TOOL_LOOP,
            model_task="tool_use",
            max_iterations=3,
            terminal_tools={"found_answer", "not_enough_info"},
        ),
        llm_request=FakeLLM(),
    )

    assert result.status == AgentStatus.SUCCESS
    assert result.output_text == "done"
    assert result.terminal_tool_name == "found_answer"
    assert observed_args == [("chat-1", {"query": "x"})]
    assert len(result.trace) >= 2
