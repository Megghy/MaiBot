from typing import Dict, List, Optional, Tuple

from src.common.data_models.database_data_model import DatabaseMessages
from src.common.data_models.info_data_model import ActionPlannerInfo
from src.config.config import global_config
from src.plugin_system.base.component_types import ActionInfo
from src.chat.replyer.base_replyer import BaseReplyer
from src.chat.replyer.prompt.replyer_prompt import (
    BASE_REPLY_SAFETY_POLICY,
    build_chat_customization_block,
    build_chat_history_block,
    build_planner_hint_block,
    build_private_identity_block,
    build_private_reply_task,
    build_private_self_reply_target,
    build_private_self_reply_task,
    build_reply_memory_block,
    build_reply_style_block,
)
from src.runtime.agent import AssembledContext, ContextAssembler


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
    ) -> Tuple[AssembledContext, List[int]]:
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

        user_parts = {
            "memory": build_reply_memory_block(
                knowledge_prompt=prompt_data["prompt_info"],
                memory_retrieval=prompt_data["memory_retrieval"],
                person_memory_block=prompt_data["person_memory_block"],
                relation_info_block=prompt_data["relation_info"],
                extra_info_block=prompt_data["extra_info_block"],
                expression_habits_block=prompt_data["expression_habits_block"],
                actions_info=prompt_data["actions_info"],
            ),
            "tool_observations": prompt_data["tool_results_block"],
            "chat_history": build_chat_history_block(
                time_block=prompt_data["time_block"],
                dialogue_prompt=prompt_data["dialogue_prompt"],
            ),
            "mood_hint": prompt_data["mood_state_prompt"],
            "chat_customization": build_chat_customization_block(
                chat_prompt_block=prompt_data["chat_prompt_block"],
                keywords_reaction_prompt=prompt_data["keywords_reaction_prompt"],
            ),
        }

        if global_config.bot.qq_account == prompt_data["user_id"] and self.chat_stream.platform == global_config.bot.platform:
            user_parts["reply_target"] = build_private_self_reply_target(
                target=prompt_data["target"],
                reason=reply_reason,
            )
            user_parts["task"] = build_private_self_reply_task()
        else:
            user_parts["reply_target"] = prompt_data["reply_target_block"]
            user_parts["planner_hint"] = build_planner_hint_block(prompt_data["planner_reasoning"])
            user_parts["task"] = build_private_reply_task(prompt_data["sender"])

        assembled = ContextAssembler().from_parts(
            system_parts={
                "policy": BASE_REPLY_SAFETY_POLICY,
                "identity": build_private_identity_block(
                    prompt_data["personality_prompt"],
                    prompt_data["sender"],
                ),
                "style": build_reply_style_block(global_config.personality.reply_style),
            },
            user_parts=user_parts,
        )

        return assembled, prompt_data["selected_expressions"]
