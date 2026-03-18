from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from src.runtime.manager import RuntimeManager
from src.runtime.models import CommandExecutionResult


@dataclass
class FakeSession:
    chat_stream: object
    stream_id: str = field(init=False)
    initialized: bool = False
    enqueued: list[tuple[object, bool]] = field(default_factory=list)
    runtime_objects: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.stream_id = self.chat_stream.stream_id

    async def initialize(self) -> None:
        self.initialized = True

    async def enqueue_prepared_message(self, message, *, persist_message: bool = False) -> None:
        self.enqueued.append((message, persist_message))

    def set_runtime_object(self, key: str, value) -> None:
        self.runtime_objects[key] = value

    def get_runtime_object(self, key: str):
        return self.runtime_objects.get(key)

    async def shutdown(self) -> None:
        return None


@pytest.mark.asyncio
async def test_runtime_manager_reuses_session(monkeypatch, fake_chat_stream):
    monkeypatch.setattr("src.runtime.manager.SessionRuntime", FakeSession)
    manager = RuntimeManager()

    first = manager.get_or_create_session(fake_chat_stream)
    second = manager.get_or_create_session(fake_chat_stream)

    assert first is second
    assert manager.get_session(fake_chat_stream.stream_id) is first
    assert manager.all_sessions()[fake_chat_stream.stream_id] is first


@pytest.mark.asyncio
async def test_runtime_manager_dispatches_message(monkeypatch, fake_chat_stream, fake_message):
    monkeypatch.setattr("src.runtime.manager.SessionRuntime", FakeSession)

    calls = {"storage": 0, "decision": 0}

    async def resolve_stream(message):
        assert message is fake_message
        return fake_chat_stream

    async def store_message(message, chat_stream):
        calls["storage"] += 1
        assert message is fake_message
        assert chat_stream is fake_chat_stream

    async def decision_runner(session, message):
        calls["decision"] += 1
        assert session.stream_id == fake_chat_stream.stream_id
        assert message is fake_message

    manager = RuntimeManager(
        decision_runner=decision_runner,
        message_storage=store_message,
        stream_resolver=resolve_stream,
    )

    session = await manager.dispatch_incoming(fake_message)

    assert session.stream_id == fake_chat_stream.stream_id
    assert fake_message.chat_stream is fake_chat_stream
    assert calls == {"storage": 1, "decision": 1}
    assert session.initialized is True


@pytest.mark.asyncio
async def test_runtime_manager_stops_on_intercepted_command(monkeypatch, fake_chat_stream, fake_message):
    monkeypatch.setattr("src.runtime.manager.SessionRuntime", FakeSession)

    calls = {"storage": 0, "decision": 0}

    async def resolve_stream(message):
        return fake_chat_stream

    async def store_message(message, chat_stream):
        calls["storage"] += 1

    async def decision_runner(session, message):
        calls["decision"] += 1

    manager = RuntimeManager(
        decision_runner=decision_runner,
        message_storage=store_message,
        stream_resolver=resolve_stream,
    )

    def matcher(text, *, message=None, session=None):
        return {"command_result": "matched"}

    async def executor(message, *, session=None, command_result=None):
        message.is_command = True
        return CommandExecutionResult(
            name="test-command",
            handled=True,
            intercepted=True,
            response=str(command_result),
        )

    manager.command_registry.register("test-command", matcher, executor, priority=100)

    session = await manager.dispatch_incoming(fake_message)

    assert calls == {"storage": 1, "decision": 0}
    assert session.enqueued == [(fake_message, False)]
