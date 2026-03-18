from __future__ import annotations

from types import MethodType, SimpleNamespace

import pytest

from src.chat.replyer.base_replyer import BaseReplyer


class DummyReplyer(BaseReplyer):
    async def build_prompt_reply_context(self, *args, **kwargs):
        raise NotImplementedError


@pytest.mark.asyncio
async def test_prepare_reply_prompt_common_context_uses_reply_time_point(monkeypatch):
    replyer = DummyReplyer.__new__(DummyReplyer)
    replyer.chat_stream = SimpleNamespace(stream_id="chat-1", platform="qq", group_info=None)
    replyer.is_group_chat = False

    timestamps = []

    def fake_fetch(*, chat_id, timestamp, limit):
        assert chat_id == "chat-1"
        timestamps.append(timestamp)
        return []

    async def fake_gather(self, chat_talking_prompt_short, sender, target, reply_reason, **kwargs):
        assert chat_talking_prompt_short == ""
        assert sender == "用户"
        assert target == "消息"
        assert reply_reason == "因为要回"
        return {
            "expression_habits_block": "",
            "selected_expressions": [1, 2],
            "tool_results_block": "",
            "prompt_info": "",
            "actions_info": "",
            "personality_prompt": "人格",
            "memory_retrieval": "",
            "mood_state_prompt": "",
            "relation_info": "",
        }

    async def fake_keywords(self, target):
        assert target == "消息"
        return "关键词提示"

    monkeypatch.setattr("src.chat.replyer.base_replyer.get_raw_msg_before_timestamp_with_chat", fake_fetch)
    monkeypatch.setattr("src.chat.replyer.base_replyer.build_person_memory_block", lambda target_person, group_id=None: "")

    replyer._gather_common_reply_blocks = MethodType(fake_gather, replyer)
    replyer.build_keywords_reaction_prompt = MethodType(fake_keywords, replyer)
    replyer._get_chat_prompt_by_type = MethodType(lambda self, chat_id, target_type: "", replyer)
    replyer.build_chat_history_prompt = MethodType(lambda self, messages: "dialogue", replyer)
    replyer._process_target_content = MethodType(
        lambda self, target, platform: (target, False, True, "", target),
        replyer,
    )
    replyer._build_reply_target_block_text = MethodType(
        lambda self, **kwargs: "reply-target",
        replyer,
    )

    result = await replyer._prepare_reply_prompt_common_context(
        extra_info="额外信息",
        reply_reason="因为要回",
        reply_time_point=123.0,
    )

    assert timestamps == [123.0, 123.0]
    assert result["dialogue_prompt"] == "dialogue"
    assert result["reply_target_block"] == "reply-target"
    assert result["keywords_reaction_prompt"] == "关键词提示"
    assert result["selected_expressions"] == [1, 2]
