from __future__ import annotations

import pytest

from src.runtime.agent import (
    AgentMode,
    AgentProfile,
    AgentRequest,
    AgentRunner,
    AgentStatus,
    PromptBlock,
    PromptBlockRole,
)


class FakeLLM:
    async def generate_response_async(self, *, prompt, tools=None, system_prompt=None):
        assert prompt == "User context"
        assert system_prompt == "System policy"
        assert tools is None
        return "final reply", ("reasoning", "fake-model", None)


@pytest.mark.asyncio
async def test_agent_runner_single_pass_returns_structured_result():
    runner = AgentRunner()
    request = AgentRequest(
        request_id="req-1",
        chat_id="chat-1",
        user_id="user-1",
        message="hello",
        prompt_blocks=[
            PromptBlock(name="policy", content="System policy", role=PromptBlockRole.SYSTEM),
            PromptBlock(name="context", content="User context", role=PromptBlockRole.USER),
        ],
    )
    profile = AgentProfile(
        name="reply-profile",
        skill_name="reply_default",
        agent_mode=AgentMode.SINGLE_PASS,
        model_task="replyer",
    )

    result = await runner.run(request, profile, llm_request=FakeLLM())

    assert result.status == AgentStatus.SUCCESS
    assert result.output_text == "final reply"
    assert result.reasoning == "reasoning"
    assert result.model_name == "fake-model"
    assert result.used_skill == "reply_default"
    assert result.system_prompt == "System policy"
    assert result.user_prompt == "User context"
    assert result.tool_calls == []
