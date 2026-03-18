from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import src.runtime.chat_controller as chat_controller_module
import src.runtime.session as session_module
from src.runtime.chat_controller import BaseChatController


class FakeChatManager:
    def __init__(self, stream):
        self._stream = stream

    def get_stream(self, stream_id: str):
        if stream_id != self._stream.stream_id:
            return None
        return self._stream

    def get_stream_name(self, stream_id: str) -> str:
        return f"name:{stream_id}"


class DummyController(BaseChatController):
    controller_name = "DummyController"

    def __init__(self, chat_id: str):
        self.seen_messages = None
        self.runtime_init_calls = 0
        super().__init__(chat_id)

    async def _initialize_runtime(self) -> None:
        self.runtime_init_calls += 1

    async def _process_recent_messages_locked(self, recent_messages_list):
        self.seen_messages = recent_messages_list
        return bool(recent_messages_list)


def _build_fake_stream(stream_id: str = "stream-1"):
    return SimpleNamespace(
        stream_id=stream_id,
        group_info=None,
        context=SimpleNamespace(get_template_name=lambda: "default"),
    )


@pytest.mark.asyncio
async def test_base_chat_controller_process_recent_messages_uses_locked_handler(monkeypatch):
    fake_stream = _build_fake_stream()
    monkeypatch.setattr(chat_controller_module, "get_chat_manager", lambda: FakeChatManager(fake_stream))

    controller = DummyController(fake_stream.stream_id)

    assert await controller.process_recent_messages() is False
    assert controller.seen_messages == []

    messages = [SimpleNamespace(message_id="1")]
    assert await controller.process_recent_messages(messages) is True
    assert controller.seen_messages == messages


@pytest.mark.asyncio
async def test_base_chat_controller_start_initializes_runtime_once(monkeypatch):
    fake_stream = _build_fake_stream()
    monkeypatch.setattr(chat_controller_module, "get_chat_manager", lambda: FakeChatManager(fake_stream))

    controller = DummyController(fake_stream.stream_id)
    loop_started = asyncio.Event()

    async def fake_main_loop():
        loop_started.set()

    controller._main_chat_loop = fake_main_loop  # type: ignore[method-assign]

    await controller.start()
    await asyncio.wait_for(loop_started.wait(), timeout=1)
    await controller._loop_task

    assert controller.runtime_init_calls == 1

    await controller.start()

    assert controller.runtime_init_calls == 1


@pytest.mark.asyncio
async def test_session_runtime_initialize_calls_controller_runtime_hook(monkeypatch):
    fake_stream = _build_fake_stream()

    class FakeController:
        def __init__(self, stream_id: str):
            self.stream_id = stream_id
            self.initialize_calls = 0

        async def initialize_runtime(self):
            self.initialize_calls += 1

    monkeypatch.setattr(session_module, "BrainChatting", FakeController)

    runtime = session_module.SessionRuntime(fake_stream)
    await runtime.initialize()

    assert runtime.controller.initialize_calls == 1
    assert runtime.get_runtime_object("heartflow_chat") is runtime.controller


def test_session_runtime_selects_private_and_group_controller(monkeypatch):
    class FakeBrainController:
        def __init__(self, stream_id: str):
            self.stream_id = stream_id
            self.kind = "brain"

        async def initialize_runtime(self):
            return None

    class FakeHeartController:
        def __init__(self, stream_id: str):
            self.stream_id = stream_id
            self.kind = "heart"
            self.last_active_time = 0.0

        async def initialize_runtime(self):
            return None

    monkeypatch.setattr(session_module, "BrainChatting", FakeBrainController)
    monkeypatch.setattr(session_module, "HeartFChatting", FakeHeartController)

    private_runtime = session_module.SessionRuntime(_build_fake_stream("private-stream"))
    group_runtime = session_module.SessionRuntime(
        SimpleNamespace(
            stream_id="group-stream",
            group_info=SimpleNamespace(group_id="1"),
        )
    )

    assert private_runtime.controller.kind == "brain"
    assert group_runtime.controller.kind == "heart"
