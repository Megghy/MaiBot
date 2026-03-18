from __future__ import annotations

import asyncio
import time
from typing import cast

from src.chat.brain_chat.brain_chat import BrainChatting
from src.chat.heart_flow.heartFC_chat import HeartFChatting
from src.chat.message_receive.chat_stream import ChatStream
from src.chat.message_receive.message import MessageRecv
from src.chat.message_receive.storage import MessageStorage
from src.chat.utils.utils import is_mentioned_bot_in_message
from src.common.data_models.database_data_model import DatabaseMessages
from src.common.logger import get_logger
from src.config.config import global_config
from src.person_info.person_info import Person
from src.runtime.chat_controller import BaseChatController
from src.runtime.models import SessionState

logger = get_logger("runtime.session")


class SessionRuntime:
    def __init__(self, chat_stream: ChatStream) -> None:
        if chat_stream.group_info:
            controller: BaseChatController = HeartFChatting(chat_stream.stream_id)
        else:
            controller = BrainChatting(chat_stream.stream_id)
        self.state = SessionState(chat_stream=chat_stream, controller=controller)
        self._flush_lock = asyncio.Lock()

    @property
    def stream_id(self) -> str:
        return self.state.chat_stream.stream_id

    @property
    def chat_stream(self) -> ChatStream:
        return self.state.chat_stream

    @property
    def controller(self) -> BaseChatController:
        return cast(BaseChatController, self.state.controller)

    def set_runtime_object(self, key: str, value) -> None:
        self.state.runtime_objects[key] = value

    def get_runtime_object(self, key: str):
        return self.state.runtime_objects.get(key)

    async def initialize(self) -> None:
        if self.state.initialized:
            return
        await self.controller.initialize_runtime()
        self.state.initialized = True
        self.set_runtime_object("heartflow_chat", self.controller)

    async def enqueue_prepared_message(self, message: MessageRecv, *, persist_message: bool = False) -> None:
        await self.initialize()

        is_mentioned, is_at, reply_probability_boost = is_mentioned_bot_in_message(message)
        message.is_mentioned = is_mentioned
        message.is_at = is_at
        message.reply_probability_boost = reply_probability_boost

        if persist_message:
            await MessageStorage.store_message(message, self.chat_stream)
        message_record = MessageStorage.build_database_message(message, self.chat_stream)
        self.state.last_message = message_record

        if not message.is_command:
            self.state.pending_messages.append(message_record)

        self._register_person(message)
        if not message.is_command:
            self._schedule_flush(message_record)

    async def handle_incoming(self, message: MessageRecv) -> None:
        await self.enqueue_prepared_message(message, persist_message=True)

    def _register_person(self, message: MessageRecv) -> None:
        user_info = message.message_info.user_info
        if not user_info or not user_info.user_id or not user_info.user_nickname:
            return
        group_id = self.chat_stream.group_info.group_id if self.chat_stream.group_info else None
        group_nick_name = user_info.user_cardname if self.chat_stream.group_info else None
        Person.register_person(
            platform=message.message_info.platform,  # type: ignore[arg-type]
            user_id=user_info.user_id,
            nickname=user_info.user_nickname,
            group_id=group_id,
            group_nick_name=group_nick_name,
        )

    def _schedule_flush(self, message_record: DatabaseMessages) -> None:
        if self.state.flush_task and not self.state.flush_task.done():
            self.state.flush_task.cancel()
        delay = self._calculate_delay(message_record)
        self.state.flush_task = asyncio.create_task(self._flush_after_delay(delay))
        self.state.flush_task.set_name(f"runtime-flush:{self.stream_id}")

    def _calculate_delay(self, message_record: DatabaseMessages) -> float:
        if not isinstance(self.controller, HeartFChatting):
            return 0.0
        if message_record.is_mentioned or message_record.is_at:
            return 0.0

        now = time.time()
        if now - self.controller.last_active_time < global_config.chat.debounce_recent_speak_window:
            return float(global_config.chat.debounce_short_idle)
        return float(global_config.chat.debounce_long_idle)

    async def _flush_after_delay(self, delay: float) -> None:
        try:
            if delay > 0:
                await asyncio.sleep(delay)
            async with self._flush_lock:
                if not self.state.pending_messages:
                    return
                pending = list(self.state.pending_messages)
                self.state.pending_messages.clear()
                await self.controller.process_recent_messages(pending)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(f"[{self.stream_id}] 处理待决消息失败: {exc}", exc_info=True)

    async def shutdown(self) -> None:
        if self.state.flush_task and not self.state.flush_task.done():
            self.state.flush_task.cancel()
            try:
                await self.state.flush_task
            except asyncio.CancelledError:
                pass
