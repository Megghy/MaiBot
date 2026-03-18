from __future__ import annotations

from typing import TYPE_CHECKING, Awaitable, Callable

from src.chat.message_receive.chat_stream import ChatStream
from src.chat.message_receive.message import MessageRecv
from src.common.logger import get_logger
from src.runtime.command_registry import CommandRegistry
from src.runtime.event_bus import EventBus

if TYPE_CHECKING:
    from src.runtime.session import SessionRuntime

logger = get_logger("runtime.manager")
SessionRuntime = None


class RuntimeManager:
    def __init__(
        self,
        *,
        decision_runner: Callable[["SessionRuntime", MessageRecv], Awaitable[None]] | None = None,
        message_storage: Callable[[MessageRecv, ChatStream], Awaitable[None]] | None = None,
        stream_resolver: Callable[[MessageRecv], Awaitable[ChatStream]] | None = None,
    ) -> None:
        self.sessions: dict[str, "SessionRuntime"] = {}
        self.decision_runner = decision_runner
        self.message_storage = message_storage
        self.stream_resolver = stream_resolver
        self.event_bus = EventBus()
        self.command_registry = CommandRegistry()

    def all_sessions(self) -> dict[str, "SessionRuntime"]:
        return self.sessions

    def get_session(self, stream_id: str) -> "SessionRuntime | None":
        return self.sessions.get(stream_id)

    def get_or_create_session(self, chat_stream: ChatStream) -> "SessionRuntime":
        global SessionRuntime
        if SessionRuntime is None:
            from src.runtime.session import SessionRuntime as SessionRuntimeClass

            SessionRuntime = SessionRuntimeClass

        session = self.sessions.get(chat_stream.stream_id)
        if session is None:
            session = SessionRuntime(chat_stream)
            self.sessions[chat_stream.stream_id] = session
        elif session.chat_stream is not chat_stream:
            session.state.chat_stream = chat_stream
        return session

    async def dispatch_incoming(self, message: MessageRecv) -> SessionRuntime:
        if not message.chat_stream:
            if self.stream_resolver is None:
                raise ValueError("消息缺少 chat_stream，且运行时未配置 stream_resolver")
            chat_stream = await self.stream_resolver(message)
            message.update_chat_stream(chat_stream)
        session = self.get_or_create_session(message.chat_stream)
        await session.initialize()

        continue_flag, updated_payload = await self.event_bus.dispatch("message.received", message)
        if not continue_flag:
            return session
        if updated_payload is not None:
            message = updated_payload

        if self.message_storage is not None:
            await self.message_storage(message, session.chat_stream)

        command_result = await self.command_registry.execute(message, session=session)
        if command_result is not None and getattr(command_result, "intercepted", False):
            await session.enqueue_prepared_message(message, persist_message=False)
            return session

        if self.decision_runner is not None:
            await self.decision_runner(session, message)
        else:
            await session.enqueue_prepared_message(message, persist_message=False)
        return session

    async def shutdown(self) -> None:
        for session in list(self.sessions.values()):
            await session.shutdown()


runtime_manager = RuntimeManager()


def set_runtime_manager(manager: RuntimeManager) -> RuntimeManager:
    global runtime_manager
    runtime_manager = manager
    return runtime_manager


def get_runtime_manager() -> RuntimeManager:
    return runtime_manager
