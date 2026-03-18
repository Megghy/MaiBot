from typing import Dict, List, Optional, Tuple

from src.common.data_models.database_data_model import DatabaseMessages
from src.common.data_models.info_data_model import ActionPlannerInfo
from src.config.config import global_config
from src.chat.utils.prompt_builder import global_prompt_manager
from src.plugin_system.base.component_types import ActionInfo
from src.chat.replyer.base_replyer import BaseReplyer


class PrivateReplyer(BaseReplyer):
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

        # 构建 System Prompt
        system_prompt = await global_prompt_manager.format_prompt(
            "private_replyer_system_prompt",
            identity=prompt_data["personality_prompt"],
            reply_style=global_config.personality.reply_style,
            moderation_prompt=prompt_data["moderation_prompt_block"],
        )

        if global_config.bot.qq_account == prompt_data["user_id"] and self.chat_stream.platform == global_config.bot.platform:
            user_prompt = await global_prompt_manager.format_prompt(
                "private_replyer_self_user_prompt",
                expression_habits_block=prompt_data["expression_habits_block"],
                tool_results_block=prompt_data["tool_results_block"],
                knowledge_prompt=prompt_data["prompt_info"],
                mood_state=prompt_data["mood_state_prompt"],
                relation_info_block=prompt_data["relation_info"],
                extra_info_block=prompt_data["extra_info_block"],
                identity=prompt_data["personality_prompt"],
                action_descriptions=prompt_data["actions_info"],
                dialogue_prompt=prompt_data["dialogue_prompt"],
                time_block=prompt_data["time_block"],
                target=prompt_data["target"],
                reason=reply_reason,
                sender_name=prompt_data["sender"],
                reply_style=global_config.personality.reply_style,
                keywords_reaction_prompt=prompt_data["keywords_reaction_prompt"],
                moderation_prompt=prompt_data["moderation_prompt_block"],
                memory_retrieval=prompt_data["memory_retrieval"],
                chat_prompt=prompt_data["chat_prompt_block"],
                person_memory_block=prompt_data["person_memory_block"],
            )
        else:
            user_prompt = await global_prompt_manager.format_prompt(
                "private_replyer_user_prompt",
                expression_habits_block=prompt_data["expression_habits_block"],
                tool_results_block=prompt_data["tool_results_block"],
                knowledge_prompt=prompt_data["prompt_info"],
                mood_state=prompt_data["mood_state_prompt"],
                relation_info_block=prompt_data["relation_info"],
                extra_info_block=prompt_data["extra_info_block"],
                action_descriptions=prompt_data["actions_info"],
                dialogue_prompt=prompt_data["dialogue_prompt"],
                time_block=prompt_data["time_block"],
                reply_target_block=prompt_data["reply_target_block"],
                keywords_reaction_prompt=prompt_data["keywords_reaction_prompt"],
                sender_name=prompt_data["sender"],
                memory_retrieval=prompt_data["memory_retrieval"],
                chat_prompt=prompt_data["chat_prompt_block"],
                planner_reasoning=prompt_data["planner_reasoning"],
                person_memory_block=prompt_data["person_memory_block"],
            )

        return user_prompt, system_prompt, prompt_data["selected_expressions"]
