from __future__ import annotations

import pytest

from src.chat.replyer import replyer_helpers


@pytest.mark.asyncio
async def test_build_external_context_blocks_short_circuits_on_memory(monkeypatch):
    calls = {"memory": 0, "knowledge": 0, "tools": 0}

    async def fake_memory(*args, **kwargs):
        calls["memory"] += 1
        return "memory-hit"

    async def fake_knowledge(*args, **kwargs):
        calls["knowledge"] += 1
        return "knowledge-hit"

    async def fake_tools(*args, **kwargs):
        calls["tools"] += 1
        return "tool-hit"

    monkeypatch.setattr("src.memory_system.memory_retrieval.build_memory_retrieval_prompt", fake_memory)
    monkeypatch.setattr(replyer_helpers, "get_prompt_info_block", fake_knowledge)
    monkeypatch.setattr(replyer_helpers, "build_tool_results_block", fake_tools)

    tool_results, prompt_info, memory_retrieval = await replyer_helpers.build_external_context_blocks(
        tool_executor=object(),
        chat_history="history",
        sender="用户",
        target="你上次说的那个呢",
        chat_stream=object(),
    )

    assert (tool_results, prompt_info, memory_retrieval) == ("", "", "memory-hit")
    assert calls == {"memory": 1, "knowledge": 0, "tools": 0}


@pytest.mark.asyncio
async def test_build_external_context_blocks_uses_knowledge_before_tools(monkeypatch):
    calls = {"memory": 0, "knowledge": 0, "tools": 0}

    async def fake_memory(*args, **kwargs):
        calls["memory"] += 1
        return ""

    async def fake_knowledge(*args, **kwargs):
        calls["knowledge"] += 1
        return "knowledge-hit"

    async def fake_tools(*args, **kwargs):
        calls["tools"] += 1
        return "tool-hit"

    monkeypatch.setattr("src.memory_system.memory_retrieval.build_memory_retrieval_prompt", fake_memory)
    monkeypatch.setattr(replyer_helpers, "get_prompt_info_block", fake_knowledge)
    monkeypatch.setattr(replyer_helpers, "build_tool_results_block", fake_tools)

    tool_results, prompt_info, memory_retrieval = await replyer_helpers.build_external_context_blocks(
        tool_executor=object(),
        chat_history="history",
        sender="用户",
        target="这是个知识类问题",
        chat_stream=object(),
    )

    assert (tool_results, prompt_info, memory_retrieval) == ("", "knowledge-hit", "")
    assert calls == {"memory": 1, "knowledge": 1, "tools": 0}
