from __future__ import annotations

from types import MethodType, SimpleNamespace

import pytest

from src.chat.replyer.group_generator import DefaultReplyer
from src.chat.replyer.private_generator import PrivateReplyer


def _build_prompt_data(*, user_id: str = "user-1", sender: str = "测试用户", target: str = "晚上好麦麦"):
    return {
        "user_id": user_id,
        "sender": sender,
        "target": target,
        "target_person": None,
        "person_memory_block": "",
        "dialogue_prompt": "[20:00] 测试用户: 晚上好麦麦",
        "planner_reasoning": "",
        "extra_info_block": "",
        "time_block": "当前时间：2026-03-19 20:00:00",
        "moderation_prompt_block": "",
        "reply_target_block": f"现在{sender}说的：{target}。引起了你的注意",
        "chat_prompt_block": "",
        "keywords_reaction_prompt": "",
        "expression_habits_block": "",
        "selected_expressions": [],
        "tool_results_block": "",
        "prompt_info": "",
        "actions_info": "",
        "personality_prompt": "是一个随手接话的人",
        "memory_retrieval": "",
        "mood_state_prompt": "",
        "relation_info": "",
    }


@pytest.mark.asyncio
async def test_group_replyer_builds_task_block_with_detected_scene():
    replyer = DefaultReplyer.__new__(DefaultReplyer)
    replyer.chat_stream = SimpleNamespace(stream_id="chat-1", platform="qq", group_info=SimpleNamespace(group_id="g1"))

    async def fake_prepare(self, **kwargs):
        return _build_prompt_data(target="晚上好麦麦")

    replyer._prepare_reply_prompt_common_context = MethodType(fake_prepare, replyer)

    assembled, _ = await replyer.build_prompt_reply_context(reply_reason="")

    task_block = next(block for block in assembled.blocks if block.name == "task")
    assert "这句像是在打招呼" in task_block.content
    assert "你说:" in task_block.content


@pytest.mark.asyncio
async def test_private_replyer_builds_task_block_with_detected_scene():
    replyer = PrivateReplyer.__new__(PrivateReplyer)
    replyer.chat_stream = SimpleNamespace(stream_id="chat-1", platform="qq")

    async def fake_prepare(self, **kwargs):
        return _build_prompt_data(target="晚饭吃什么好")

    replyer._prepare_reply_prompt_common_context = MethodType(fake_prepare, replyer)

    assembled, _ = await replyer.build_prompt_reply_context(reply_reason="")

    task_block = next(block for block in assembled.blocks if block.name == "task")
    assert "这句像是在提问" in task_block.content
    assert "你说:" in task_block.content
