import time
import asyncio
import random
from typing import List, Optional, Dict, Tuple, Any
from datetime import datetime

from src.common.logger import get_logger
from src.common.data_models.database_data_model import DatabaseMessages
from src.common.data_models.info_data_model import ActionPlannerInfo
from src.config.config import global_config
from src.chat.message_receive.message import UserInfo, Seg, MessageRecv, MessageSending
from src.chat.utils.prompt_builder import global_prompt_manager
from src.chat.utils.chat_message_builder import (
    build_readable_messages,
    get_raw_msg_before_timestamp_with_chat,
    replace_user_references,
)
from src.person_info.person_info import Person
from src.chat.replyer.replyer_utils import build_person_memory_block
from src.plugin_system.base.component_types import ActionInfo
from src.chat.replyer.base_replyer import BaseReplyer
from src.express.expression_selector import expression_selector

logger = get_logger("replyer")


class DefaultReplyer(BaseReplyer):
    def build_chat_history_prompt(
        self, message_list_before_now: List[DatabaseMessages]
    ) -> str:
        """构建群聊背景对话 prompt"""

        if not message_list_before_now:
            return ""

        latest_msgs = message_list_before_now[-int(global_config.chat.max_context_size) :]
        return build_readable_messages(
            latest_msgs,
            replace_bot_name=True,
            timestamp_mode="normal_no_YMD",
            truncate=True,
        )

    async def build_prompt_reply_context(
        self,
        reply_message: Optional[DatabaseMessages] = None,
        extra_info: str = "",
        reply_reason: str = "",
        available_actions: Optional[Dict[str, ActionInfo]] = None,
        chosen_actions: Optional[List[ActionPlannerInfo]] = None,
        enable_tool: bool = True,
        reply_time_point: Optional[float] = None,
    ) -> Tuple[str, str, List[int]]:
        """
        构建回复器上下文
        """
        if reply_time_point is None:
            reply_time_point = time.time()

        if available_actions is None:
            available_actions = {}
        chat_stream = self.chat_stream
        chat_id = chat_stream.stream_id
        platform = chat_stream.platform
        group_id: Optional[str] = None
        if chat_stream.group_info:
            try:
                group_id = chat_stream.group_info.group_id  # type: ignore[attr-defined]
            except Exception:
                group_id = None

        user_id = "用户ID"
        person_name = "用户"
        sender = "用户"
        target = "消息"

        target_person: Optional[Person] = None
        if reply_message:
            user_id = reply_message.user_info.user_id
            person = Person(platform=platform, user_id=user_id)
            target_person = person
            person_name = person.person_name or user_id
            sender = person_name
            target = reply_message.processed_plain_text

        (
            target,
            has_only_pics,
            has_text,
            pic_part,
            text_part,
        ) = self._process_target_content(target, chat_stream.platform)

        person_memory_block = build_person_memory_block(target_person, group_id=group_id)

        message_list_before_now_long = get_raw_msg_before_timestamp_with_chat(
            chat_id=chat_id,
            timestamp=reply_time_point,
            limit=global_config.chat.max_context_size * 1,
        )

        message_list_before_short = get_raw_msg_before_timestamp_with_chat(
            chat_id=chat_id,
            timestamp=reply_time_point,
            limit=int(global_config.chat.max_context_size * 0.33),
        )

        person_list_short: List[Person] = []
        for msg in message_list_before_short:
            if (
                global_config.bot.qq_account == msg.user_info.user_id
                and global_config.bot.platform == msg.user_info.platform
            ):
                continue
            if (
                reply_message
                and reply_message.user_info.user_id == msg.user_info.user_id
                and reply_message.user_info.platform == msg.user_info.platform
            ):
                continue
            person = Person(platform=msg.user_info.platform, user_id=msg.user_info.user_id)
            if person.is_known:
                person_list_short.append(person)

        chat_talking_prompt_short = build_readable_messages(
            message_list_before_short,
            replace_bot_name=True,
            timestamp_mode="relative",
            read_mark=0.0,
            show_actions=True,
        )

        gather_results = await self._gather_common_reply_blocks(
            chat_talking_prompt_short,
            sender,
            target,
            reply_reason,
            available_actions=available_actions,
            chosen_actions=chosen_actions,
            enable_tool=enable_tool,
        )

        expression_habits_block = gather_results["expression_habits_block"]
        selected_expressions = gather_results["selected_expressions"]
        tool_results_block: str = gather_results["tool_results_block"]
        prompt_info: str = gather_results["prompt_info"]
        actions_info: str = gather_results["actions_info"]
        personality_prompt: str = gather_results["personality_prompt"]
        memory_retrieval: str = gather_results["memory_retrieval"]
        mood_state_prompt: str = gather_results["mood_state_prompt"]
        keywords_reaction_prompt = await self.build_keywords_reaction_prompt(target)

        # 从 chosen_actions 中提取 planner 的整体思考理由
        planner_reasoning = ""
        if global_config.chat.include_planner_reasoning and reply_reason:
            # 如果没有 chosen_actions，使用 reply_reason 作为备选
            planner_reasoning = f"你的想法是：{reply_reason}"

        if extra_info:
            extra_info_block = f"以下是你在回复时需要参考的信息，现在请你阅读以下内容，进行决策\n{extra_info}\n以上是你在回复时需要参考的信息，现在请你阅读以下内容，进行决策"
        else:
            extra_info_block = ""

        time_block = f"当前时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        moderation_prompt_block = "请不要输出违法违规内容，不要输出色情，暴力，政治相关内容，如有敏感内容，请规避。"

        reply_target_block = self._build_reply_target_block_text(
            is_group_chat=True,
            sender=sender,
            target=target,
            has_only_pics=has_only_pics,
            has_text=has_text,
            pic_part=pic_part,
            text_part=text_part,
        )

        # 构建分离的对话 prompt
        dialogue_prompt = self.build_chat_history_prompt(message_list_before_now_long)

        # 获取匹配的额外prompt
        chat_prompt_content = self._get_chat_prompt_by_type(chat_id, "group")
        chat_prompt_block = f"{chat_prompt_content}\n" if chat_prompt_content else ""

        # 固定使用群聊回复模板
        system_prompt = await global_prompt_manager.format_prompt(
            "replyer_system_prompt",
            identity=personality_prompt,
            bot_name=global_config.bot.nickname,
            reply_style=global_config.personality.reply_style,
        )

        user_prompt = await global_prompt_manager.format_prompt(
            "replyer_user_prompt",
            expression_habits_block=expression_habits_block,
            tool_results_block=tool_results_block,
            knowledge_prompt=prompt_info,
            mood_state=mood_state_prompt,
            # relation_info_block=relation_info,
            extra_info_block=extra_info_block,
            # identity=personality_prompt,
            action_descriptions=actions_info,
            sender_name=sender,
            dialogue_prompt=dialogue_prompt,
            time_block=time_block,
            reply_target_block=reply_target_block,
            # reply_style=global_config.personality.reply_style,
            keywords_reaction_prompt=keywords_reaction_prompt,
            moderation_prompt=moderation_prompt_block,
            memory_retrieval=memory_retrieval,
            chat_prompt=chat_prompt_block,
            planner_reasoning=planner_reasoning,
            person_memory_block=person_memory_block,
        )

        return user_prompt, system_prompt, selected_expressions


def weighted_sample_no_replacement(items, weights, k) -> list:
    """
    加权且不放回地随机抽取k个元素。
    """
    selected = []
    pool = list(zip(items, weights, strict=False))
    for _ in range(min(k, len(pool))):
        total = sum(w for _, w in pool)
        r = random.uniform(0, total)
        upto = 0
        for idx, (item, weight) in enumerate(pool):
            upto += weight
            if upto >= r:
                selected.append(item)
                pool.pop(idx)
                break
    return selected
