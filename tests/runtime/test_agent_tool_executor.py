from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.runtime.agent import AgentResult, AgentStatus, ToolCallRecord
from src.runtime.agent.tool_executor import AgentToolExecutor


@pytest.mark.asyncio
async def test_tool_executor_uses_structured_blocks(monkeypatch):
    chat_manager = SimpleNamespace(
        get_stream=lambda chat_id: SimpleNamespace(stream_id=chat_id),
        get_stream_name=lambda chat_id: "",
    )
    monkeypatch.setattr("src.runtime.agent.tool_executor.get_chat_manager", lambda: chat_manager)
    monkeypatch.setattr(
        "src.runtime.agent.tool_executor.get_llm_available_tool_definitions",
        lambda: [("search_tool", {"name": "search_tool"})],
    )

    seen: dict[str, object] = {}

    async def fake_run(self, request, profile, **kwargs):
        seen["request"] = request
        seen["profile"] = profile
        seen["kwargs"] = kwargs
        return AgentResult(
            status=AgentStatus.SUCCESS,
            tool_calls=[ToolCallRecord(call_id="call-1", tool_name="search_tool", args={"query": "x"})],
            used_skill=profile.skill_name,
        )

    async def fake_execute_tool_calls(self, tool_calls):
        assert len(tool_calls) == 1
        assert tool_calls[0].func_name == "search_tool"
        return ([{"tool_name": "search_tool", "content": "ok"}], ["search_tool"])

    monkeypatch.setattr("src.runtime.agent.tool_executor.AgentRunner.run", fake_run)
    monkeypatch.setattr("src.runtime.agent.tool_executor.AgentToolExecutor.execute_tool_calls", fake_execute_tool_calls)

    executor = AgentToolExecutor(chat_id="chat-1")
    tool_results, used_tools, prompt = await executor.execute_from_chat_message(
        target_message="帮我查一下天气",
        chat_history="A: 今天天气怎么样",
        sender="Alice",
        return_details=True,
    )

    assert tool_results == [{"tool_name": "search_tool", "content": "ok"}]
    assert used_tools == ["search_tool"]
    assert "聊天记录:" in prompt
    assert "Alice 刚刚发送的内容是: 帮我查一下天气" in prompt
    assert "任务:" in prompt

    request = seen["request"]
    block_names = [block.name for block in request.prompt_blocks]
    assert block_names == ["policy", "identity", "chat_history", "reply_target", "task", "output_contract"]

    profile = seen["profile"]
    assert profile.name == "tool_only"
    assert profile.skill_name == "tool_only"

