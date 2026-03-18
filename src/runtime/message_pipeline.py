from __future__ import annotations

from typing import Any, Awaitable, Callable

from src.chat.message_receive.chat_stream import ChatManager, ChatStream, get_chat_manager
from src.chat.message_receive.message import MessageRecv
from src.runtime.manager import RuntimeManager

NoticeHandler = Callable[[MessageRecv], Awaitable[bool | None]]
BanWordChecker = Callable[[str, Any, Any | None], bool]
BanRegexChecker = Callable[[str, Any, Any | None], bool]


class MessagePipeline:
    """Runtime-facing receive pipeline.

    ChatBot keeps transport-boundary concerns, while active orchestration lives here.
    """

    def __init__(
        self,
        runtime: RuntimeManager,
        *,
        chat_manager: ChatManager | None = None,
        notice_handler: NoticeHandler | None = None,
        ban_word_checker: BanWordChecker | None = None,
        ban_regex_checker: BanRegexChecker | None = None,
    ) -> None:
        self.runtime = runtime
        self.chat_manager = chat_manager or get_chat_manager()
        self.notice_handler = notice_handler
        self.ban_word_checker = ban_word_checker
        self.ban_regex_checker = ban_regex_checker

    def normalize_message_data(self, message_data: dict[str, Any]) -> None:
        message_info = message_data["message_info"]
        group_info = message_info.get("group_info")
        if group_info is not None:
            group_info["group_id"] = str(group_info["group_id"])

        user_info = message_info.get("user_info")
        if user_info is not None:
            user_info["user_id"] = str(user_info["user_id"])

    async def resolve_chat_stream(self, message: MessageRecv) -> ChatStream:
        return await self.chat_manager.resolve_stream(message)

    async def prepare_message(self, message_data: dict[str, Any]) -> MessageRecv | None:
        self.normalize_message_data(message_data)
        message = MessageRecv(message_data)

        if self.notice_handler is not None:
            await self.notice_handler(message)

        await message.process()

        user_info = message.message_info.user_info
        group_info = message.message_info.group_info
        if user_info is not None:
            if self.ban_word_checker and self.ban_word_checker(message.processed_plain_text, user_info, group_info):
                return None
            if self.ban_regex_checker and self.ban_regex_checker(message.raw_message, user_info, group_info):
                return None

        return message

    async def process(self, message_data: dict[str, Any]):
        message = await self.prepare_message(message_data)
        if message is None:
            return None
        return await self.runtime.dispatch_incoming(message)
