from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.chat.replyer.base_replyer import BaseReplyer
from src.runtime.agent import AgentResult, AgentStatus, PromptBlockRole, ToolCallRecord


@pytest.mark.asyncio
async def test_reply_llm_generate_content_routes_through_agent_runner(monkeypatch):
    replyer = BaseReplyer.__new__(BaseReplyer)
    replyer.chat_stream = SimpleNamespace(stream_id="chat-1")
    replyer.express_model = object()

    seen = {}

    async def fake_run(self, request, profile, **kwargs):
        seen["request"] = request
        seen["profile"] = profile
        seen["kwargs"] = kwargs
        return AgentResult(
            status=AgentStatus.SUCCESS,
            output_text="reply text",
            reasoning="reasoning text",
            model_name="fake-model",
            used_skill=profile.skill_name,
        )

    monkeypatch.setattr("src.chat.replyer.base_replyer.AgentRunner.run", fake_run)

    content, reasoning, model_name, tool_calls = await replyer.llm_generate_content(
        "User prompt",
        system_prompt="System prompt",
        allow_refuse=True,
        reasoning_config={"enabled": True},
    )

    assert content == "reply text"
    assert reasoning == "reasoning text"
    assert model_name == "fake-model"
    assert tool_calls == []

    request = seen["request"]
    assert request.chat_id == "chat-1"
    assert request.metadata["reasoning_config"] == {"enabled": True}
    assert request.prompt_blocks[0].role == PromptBlockRole.SYSTEM
    assert request.prompt_blocks[0].content == "System prompt"
    assert request.prompt_blocks[1].role == PromptBlockRole.USER
    assert request.prompt_blocks[1].content == "User prompt"
    assert any(block.name == "output_contract" for block in request.prompt_blocks)

    profile = seen["profile"]
    assert profile.name == "reply_default"
    assert profile.model_task == "replyer"


@pytest.mark.asyncio
async def test_reply_llm_generate_content_honors_refuse_tool(monkeypatch):
    replyer = BaseReplyer.__new__(BaseReplyer)
    replyer.chat_stream = SimpleNamespace(stream_id="chat-1")
    replyer.express_model = object()

    async def fake_run(self, request, profile, **kwargs):
        return AgentResult(
            status=AgentStatus.SUCCESS,
            output_text="should be cleared",
            reasoning="base reasoning",
            model_name="fake-model",
            tool_calls=[
                ToolCallRecord(
                    call_id="call-1",
                    tool_name="refuse_to_reply",
                    args={"reason": "too risky"},
                )
            ],
            used_skill=profile.skill_name,
        )

    monkeypatch.setattr("src.chat.replyer.base_replyer.AgentRunner.run", fake_run)

    content, reasoning, model_name, tool_calls = await replyer.llm_generate_content(
        "User prompt",
        system_prompt="System prompt",
        allow_refuse=True,
    )

    assert content == ""
    assert model_name == "fake-model"
    assert len(tool_calls) == 1
    assert tool_calls[0].func_name == "refuse_to_reply"
    assert "[System] Refused to reply: too risky" in reasoning
