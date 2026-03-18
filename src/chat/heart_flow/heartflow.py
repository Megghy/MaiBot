import traceback
from typing import Any, Optional

from src.chat.message_receive.chat_stream import get_chat_manager
from src.common.logger import get_logger
from src.chat.heart_flow.heartFC_chat import HeartFChatting
from src.chat.brain_chat.brain_chat import BrainChatting
from src.chat.message_receive.chat_stream import ChatStream
from src.runtime.manager import get_runtime_manager

logger = get_logger("heartflow")


class Heartflow:
    """主心流协调器，负责初始化并协调聊天"""

    @property
    def heartflow_chat_list(self) -> dict[Any, HeartFChatting | BrainChatting]:
        chats: dict[Any, HeartFChatting | BrainChatting] = {}
        for stream_id, session in get_runtime_manager().all_sessions().items():
            if chat := session.get_runtime_object("heartflow_chat"):
                chats[stream_id] = chat
        return chats

    async def get_or_create_heartflow_chat(self, chat_id: Any) -> Optional[HeartFChatting | BrainChatting]:
        """获取或创建一个新的HeartFChatting实例"""
        try:
            runtime_manager = get_runtime_manager()
            session = runtime_manager.get_session(chat_id)
            if session and (chat := session.get_runtime_object("heartflow_chat")):
                return chat

            chat_stream: ChatStream | None = get_chat_manager().get_stream(chat_id)
            if not chat_stream:
                raise ValueError(f"未找到 chat_id={chat_id} 的聊天流")

            session = session or runtime_manager.get_or_create_session(chat_stream)
            await session.initialize()
            if chat := session.get_runtime_object("heartflow_chat"):
                return chat
            session.set_runtime_object("heartflow_chat", session.controller)
            return session.controller
        except Exception as e:
            logger.error(f"创建心流聊天 {chat_id} 失败: {e}", exc_info=True)
            traceback.print_exc()
            return None


heartflow = Heartflow()
