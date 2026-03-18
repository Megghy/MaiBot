from typing import Dict, List, Optional, Tuple

from src.common.data_models.database_data_model import DatabaseMessages
from src.common.data_models.info_data_model import ActionPlannerInfo
from src.config.config import global_config
from src.chat.utils.prompt_builder import global_prompt_manager
from src.chat.utils.chat_message_builder import build_readable_messages
from src.plugin_system.base.component_types import ActionInfo
from src.chat.replyer.base_replyer import BaseReplyer


class DefaultReplyer(BaseReplyer):
    def build_chat_history_prompt(self, message_list_before_now: List[DatabaseMessages]) -> str:
        """构建群聊背景对话 prompt"""

        if not message_list_before_now:
            return ""

        latest_msgs = message_list_before_now[-int(global_config.chat.max_context_size) :]
        return build_readable_messages(
            latest_msgs,
            replace_bot_name=True,
            timestamp_mode="normal_no_YMD",
            truncate=True,
            show_pic_mapping_header=False,
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
        prompt_data = await self._prepare_reply_prompt_common_context(
            reply_message=reply_message,
            extra_info=extra_info,
            reply_reason=reply_reason,
            available_actions=available_actions,
            chosen_actions=chosen_actions,
            enable_tool=enable_tool,
            reply_time_point=reply_time_point,
        )

        # 固定使用群聊回复模板
        system_prompt = await global_prompt_manager.format_prompt(
            "replyer_system_prompt",
            identity=prompt_data["personality_prompt"],
            bot_name=global_config.bot.nickname,
            reply_style=global_config.personality.reply_style,
        )

        user_prompt = await global_prompt_manager.format_prompt(
            "replyer_user_prompt",
            expression_habits_block=prompt_data["expression_habits_block"],
            tool_results_block=prompt_data["tool_results_block"],
            knowledge_prompt=prompt_data["prompt_info"],
            mood_state=prompt_data["mood_state_prompt"],
            relation_info_block=prompt_data["relation_info"],
            extra_info_block=prompt_data["extra_info_block"],
            action_descriptions=prompt_data["actions_info"],
            sender_name=prompt_data["sender"],
            dialogue_prompt=prompt_data["dialogue_prompt"],
            time_block=prompt_data["time_block"],
            reply_target_block=prompt_data["reply_target_block"],
            keywords_reaction_prompt=prompt_data["keywords_reaction_prompt"],
            moderation_prompt=prompt_data["moderation_prompt_block"],
            memory_retrieval=prompt_data["memory_retrieval"],
            chat_prompt=prompt_data["chat_prompt_block"],
            planner_reasoning=prompt_data["planner_reasoning"],
            person_memory_block=prompt_data["person_memory_block"],
        )

        return user_prompt, system_prompt, prompt_data["selected_expressions"]
