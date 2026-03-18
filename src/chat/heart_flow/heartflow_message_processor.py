import re
import traceback

from typing import TYPE_CHECKING

from src.chat.message_receive.message import MessageRecv
from src.chat.message_receive.storage import MessageStorage
from src.chat.heart_flow.heartflow import heartflow
from src.chat.utils.utils import is_mentioned_bot_in_message
from src.chat.utils.chat_message_builder import replace_user_references
from src.common.logger import get_logger
from src.common.data_models.database_data_model import DatabaseMessages
from src.person_info.person_info import Person
from src.common.database.database_model import Images
from src.runtime.manager import get_runtime_manager

if TYPE_CHECKING:
    pass

logger = get_logger("chat")


class HeartFCMessageReceiver:
    """心流处理器，负责处理接收到的消息并计算兴趣度"""

    def __init__(self):
        """初始化心流处理器，创建消息存储实例"""
        self.storage = MessageStorage()

    @staticmethod
    def _to_runtime_message(message: MessageRecv) -> DatabaseMessages:
        chat = message.chat_stream
        if chat is None:
            raise ValueError("message.chat_stream is required")

        user_info = message.message_info.user_info
        group_info = chat.group_info
        chat_user_info = chat.user_info

        return DatabaseMessages(
            message_id=str(message.message_info.message_id),
            time=float(message.message_info.time),  # type: ignore[arg-type]
            chat_id=chat.stream_id,
            reply_to="",
            interest_value=message.interest_value,
            key_words="",
            key_words_lite="",
            is_mentioned=bool(message.is_mentioned),
            is_at=bool(message.is_at),
            reply_probability_boost=message.reply_probability_boost,
            processed_plain_text=message.processed_plain_text,
            display_message="",
            priority_mode=message.priority_mode,
            priority_info=message.priority_info,
            is_emoji=message.is_emoji,
            is_picid=message.is_picid,
            is_command=message.is_command,
            is_notify=message.is_notify,
            selected_expressions="",
            user_id=str(user_info.user_id) if user_info else "",
            user_nickname=user_info.user_nickname if user_info else "",
            user_cardname=user_info.user_cardname if user_info else None,
            user_platform=user_info.platform if user_info else "",
            chat_info_group_id=str(group_info.group_id) if group_info else None,
            chat_info_group_name=group_info.group_name if group_info else None,
            chat_info_group_platform=group_info.platform if group_info else None,
            chat_info_user_id=str(chat_user_info.user_id) if chat_user_info else "",
            chat_info_user_nickname=chat_user_info.user_nickname if chat_user_info else "",
            chat_info_user_cardname=chat_user_info.user_cardname if chat_user_info else None,
            chat_info_user_platform=chat_user_info.platform if chat_user_info else "",
            chat_info_stream_id=chat.stream_id,
            chat_info_platform=chat.platform,
            chat_info_create_time=float(chat.create_time),
            chat_info_last_active_time=float(chat.last_active_time),
        )

    async def process_message(self, message: MessageRecv, *, persist_message: bool = True) -> None:
        """处理接收到的原始消息数据

        主要流程:
        1. 消息解析与初始化
        2. 消息缓冲处理
        3. 过滤检查
        4. 兴趣度计算
        5. 关系处理

        Args:
            message_data: 原始消息字符串
        """
        try:
            # 1. 消息解析与初始化
            userinfo = message.message_info.user_info
            user_nickname = userinfo.user_nickname if userinfo else "未知用户"
            user_cardname = userinfo.user_cardname if userinfo else None
            user_id = userinfo.user_id if userinfo else None
            chat = message.chat_stream

            # 2. 计算at信息
            is_mentioned, is_at, reply_probability_boost = is_mentioned_bot_in_message(message)
            # print(f"is_mentioned: {is_mentioned}, is_at: {is_at}, reply_probability_boost: {reply_probability_boost}")
            message.is_mentioned = is_mentioned
            message.is_at = is_at
            message.reply_probability_boost = reply_probability_boost

            runtime_manager = get_runtime_manager()
            session = runtime_manager.get_session(chat.stream_id) if chat else None  # type: ignore[union-attr]
            if chat and (session is not None or runtime_manager is not None):
                session = session or runtime_manager.get_or_create_session(chat)
                await session.initialize()
                await session.enqueue_prepared_message(message, persist_message=persist_message)
                return

            if persist_message:
                await self.storage.store_message(message, chat)

            if active_chat := await heartflow.get_or_create_heartflow_chat(chat.stream_id):  # type: ignore[arg-type]
                runtime_message = self._to_runtime_message(message)
                await active_chat.process_recent_messages([runtime_message])

            # 3. 日志记录
            mes_name = chat.group_info.group_name if chat.group_info else "私聊"

            # 用这个pattern截取出id部分，picid是一个list，并替换成对应的图片描述
            picid_pattern = r"\[picid:([^\]]+)\]"
            picid_list = re.findall(picid_pattern, message.processed_plain_text)

            # 创建替换后的文本
            processed_text = message.processed_plain_text
            if picid_list:
                for picid in picid_list:
                    image = Images.get_or_none(Images.image_id == picid)
                    if image and image.description:
                        # 将[picid:xxxx]替换成图片描述
                        processed_text = processed_text.replace(f"[picid:{picid}]", f"[图片：{image.description}]")
                    else:
                        # 如果没有找到图片描述，则移除[picid:xxxx]标记
                        processed_text = processed_text.replace(f"[picid:{picid}]", "[图片：网络不好，图片无法加载]")

            # 应用用户引用格式替换，将回复<aaa:bbb>和@<aaa:bbb>格式转换为可读格式
            processed_plain_text = replace_user_references(
                processed_text,
                message.message_info.platform,  # type: ignore
                replace_bot_name=True,
            )
            # if not processed_plain_text:
            # print(message)

            logger.info(f"[{mes_name}]{user_nickname}:{processed_plain_text}")

            # 如果是群聊，获取群号和群昵称
            group_id = None
            group_nick_name = None
            if chat.group_info:
                group_id = chat.group_info.group_id  # type: ignore
                group_nick_name = user_cardname

            if user_id and user_nickname:
                _ = Person.register_person(
                    platform=message.message_info.platform,  # type: ignore
                    user_id=user_id,
                    nickname=user_nickname,
                    group_id=group_id,
                    group_nick_name=group_nick_name,
                )

        except Exception as e:
            logger.error(f"消息处理失败: {e}")
            print(traceback.format_exc())
