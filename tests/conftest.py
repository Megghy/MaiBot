from __future__ import annotations

from types import SimpleNamespace

import pytest


class FakeMessage:
    def __init__(self, text: str = "hello") -> None:
        self.processed_plain_text = text
        self.chat_stream = None
        self.is_command = False
        self.updated_stream = None

    def update_chat_stream(self, chat_stream) -> None:
        self.chat_stream = chat_stream
        self.updated_stream = chat_stream


@pytest.fixture
def fake_chat_stream():
    return SimpleNamespace(
        stream_id="stream-1",
        group_info=None,
        user_info=SimpleNamespace(user_id="u1", user_nickname="tester", user_cardname=None, platform="qq"),
        platform="qq",
    )


@pytest.fixture
def fake_message():
    return FakeMessage()
