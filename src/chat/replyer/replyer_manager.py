from typing import Optional

from src.common.logger import get_logger
from src.chat.message_receive.chat_stream import ChatStream, get_chat_manager
from src.chat.replyer.base_replyer import BaseReplyer
from src.chat.replyer.group_generator import DefaultReplyer
from src.chat.replyer.private_generator import PrivateReplyer
from src.runtime.manager import get_runtime_manager

logger = get_logger("ReplyerManager")


class ReplyerManager:
    def get_replyer(
        self,
        chat_stream: Optional[ChatStream] = None,
        chat_id: Optional[str] = None,
        request_type: str = "replyer",
    ) -> Optional[BaseReplyer]:
        """
        获取或创建回复器实例。

        model_configs 仅在首次为某个 chat_id/stream_id 创建实例时有效。
        后续调用将返回已缓存的实例，忽略 model_configs 参数。
        """
        stream_id = chat_stream.stream_id if chat_stream else chat_id
        if not stream_id:
            logger.warning("[ReplyerManager] 缺少 stream_id，无法获取回复器。")
            return None

        runtime_manager = get_runtime_manager()
        session = runtime_manager.get_session(stream_id)
        if session and (replyer := session.get_runtime_object("replyer")):
            logger.debug(f"[ReplyerManager] 为 stream_id '{stream_id}' 返回已存在的回复器实例。")
            replyer.chat_stream = session.chat_stream
            return replyer

        logger.debug(f"[ReplyerManager] 为 stream_id '{stream_id}' 创建新的回复器实例并缓存。")

        target_stream = chat_stream or (session.chat_stream if session else None)
        if not target_stream:
            if chat_manager := get_chat_manager():
                target_stream = chat_manager.get_stream(stream_id)

        if not target_stream:
            logger.warning(f"[ReplyerManager] 未找到 stream_id='{stream_id}' 的聊天流，无法创建回复器。")
            return None

        # model_configs 只在此时（初始化时）生效
        if target_stream.group_info:
            replyer = DefaultReplyer(
                chat_stream=target_stream,
                request_type=request_type,
            )
        else:
            replyer = PrivateReplyer(
                chat_stream=target_stream,
                request_type=request_type,
            )

        session = session or runtime_manager.get_or_create_session(target_stream)
        session.set_runtime_object("replyer", replyer)
        return replyer


# 创建一个全局实例
replyer_manager = ReplyerManager()
