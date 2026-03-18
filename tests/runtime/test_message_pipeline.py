from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.runtime.message_pipeline import MessagePipeline


class FakePipelineMessage:
    def __init__(self, data):
        message_info = data["message_info"]
        group_info = message_info.get("group_info")
        user_info = message_info.get("user_info")

        self.message_info = SimpleNamespace(
            group_info=SimpleNamespace(**group_info) if group_info else None,
            user_info=SimpleNamespace(**user_info) if user_info else None,
        )
        self.raw_message = data["raw_message"]
        self.processed_plain_text = ""
        self.chat_stream = None

    async def process(self) -> None:
        self.processed_plain_text = self.raw_message


@pytest.mark.asyncio
async def test_message_pipeline_processes_message(monkeypatch):
    monkeypatch.setattr("src.runtime.message_pipeline.MessageRecv", FakePipelineMessage)

    calls = {"notice": 0, "dispatch": 0}

    async def dispatch_incoming(message):
        calls["dispatch"] += 1
        return message

    async def notice_handler(message):
        calls["notice"] += 1
        return False

    runtime = SimpleNamespace(dispatch_incoming=dispatch_incoming)
    pipeline = MessagePipeline(
        runtime,
        notice_handler=notice_handler,
        ban_word_checker=lambda text, user_info, group_info: False,
        ban_regex_checker=lambda text, user_info, group_info: False,
    )

    payload = {
        "raw_message": "hello",
        "message_info": {
            "group_info": {"group_id": 123},
            "user_info": {"user_id": 456, "user_nickname": "tester"},
        },
    }

    message = await pipeline.process(payload)

    assert calls == {"notice": 1, "dispatch": 1}
    assert payload["message_info"]["group_info"]["group_id"] == "123"
    assert payload["message_info"]["user_info"]["user_id"] == "456"
    assert message.processed_plain_text == "hello"


@pytest.mark.asyncio
async def test_message_pipeline_drops_filtered_message(monkeypatch):
    monkeypatch.setattr("src.runtime.message_pipeline.MessageRecv", FakePipelineMessage)

    calls = {"dispatch": 0}

    async def dispatch_incoming(message):
        calls["dispatch"] += 1
        return message

    runtime = SimpleNamespace(dispatch_incoming=dispatch_incoming)
    pipeline = MessagePipeline(
        runtime,
        ban_word_checker=lambda text, user_info, group_info: text == "blocked",
        ban_regex_checker=lambda text, user_info, group_info: False,
    )

    payload = {
        "raw_message": "blocked",
        "message_info": {
            "group_info": None,
            "user_info": {"user_id": 456, "user_nickname": "tester"},
        },
    }

    result = await pipeline.process(payload)

    assert result is None
    assert calls["dispatch"] == 0
