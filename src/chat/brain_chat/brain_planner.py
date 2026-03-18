from __future__ import annotations

import time
import traceback
from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from rich.traceback import install

from src.chat.message_receive.chat_stream import get_chat_manager
from src.chat.planner_actions.action_manager import ActionManager
from src.chat.planner_actions.planner_runtime import (
    PreparedPlannerExecution,
    collect_planner_chat_context,
    execute_planner_llm,
)
from src.chat.planner_actions.planner_selection import (
    filter_actions_by_activation_type,
    get_necessary_info as resolve_necessary_info,
)
from src.chat.utils.chat_message_builder import (
    build_readable_actions,
    get_actions_by_timestamp_with_chat,
)
from src.common.data_models.info_data_model import ActionPlannerInfo
from src.common.logger import get_logger
from src.config.config import global_config, model_config
from src.llm_models.utils_model import LLMRequest
from src.plugin_system.base.component_types import ActionInfo
from src.runtime.agent import ContextAssembler, PromptBlock, PromptComposer, build_agent_profile

if TYPE_CHECKING:
    from src.common.data_models.database_data_model import DatabaseMessages
    from src.common.data_models.info_data_model import TargetPersonInfo

logger = get_logger("planner")

install(extra_lines=3)


def init_prompt() -> None:
    return None


def _join_non_empty(*parts: str) -> str:
    normalized_parts = [part.strip() for part in parts if part and part.strip()]
    return "\n\n".join(normalized_parts)


def _build_brain_planner_policy_block() -> str:
    return _join_non_empty(
        "请不要输出违法违规内容，不要输出色情、暴力、政治相关内容，如有敏感内容请规避。",
        global_config.personality.private_plan_style,
    )


def _build_brain_planner_identity_block(interest: str) -> str:
    bot_name = global_config.bot.nickname
    bot_nickname = f"，也可以叫你{','.join(global_config.bot.alias_names)}" if global_config.bot.alias_names else ""
    return "\n".join(
        [
            f"你的名字是{bot_name}{bot_nickname}。",
            "你正在负责为当前聊天选择下一步动作。",
            "请注意区分哪些发言是你自己的发言。",
            f"你的兴趣是: {interest}",
        ]
    )


def _build_brain_planner_state_block(
    *,
    time_block: str,
    chat_context_description: str,
    actions_before_now_block: str,
) -> str:
    return _join_non_empty(
        time_block,
        chat_context_description,
        actions_before_now_block,
    )


def _build_brain_planner_task_block() -> str:
    return "\n".join(
        [
            "任务清单:",
            "1. 阅读最新聊天记录，判断当前是否需要行动。",
            "2. 结合最近已经执行过的 action，避免重复操作。",
            "3. 只允许通过 tool calls 表达决定，不要输出 JSON、自然语言总结或其他自由文本。",
            "4. 如果不需要执行任何动作，请调用 `no_reply` 工具并说明原因。",
        ]
    )


class BrainPlanner:
    def __init__(self, chat_id: str, action_manager: ActionManager):
        self.chat_id = chat_id
        self.log_prefix = f"[{get_chat_manager().get_stream_name(chat_id) or chat_id}]"
        self.action_manager = action_manager
        self.planner_llm = LLMRequest(model_set=model_config.model_task_config.planner, request_type="planner")
        self.last_obs_time_mark = 0.0

    async def plan(
        self,
        available_actions: Dict[str, ActionInfo],
        loop_start_time: float = 0.0,
    ) -> List[ActionPlannerInfo]:
        prepared_plan = await self.prepare_plan_context(available_actions)
        return await self.execute_prepared_plan(
            prepared_plan=prepared_plan,
            available_actions=available_actions,
            loop_start_time=loop_start_time,
        )

    async def prepare_plan_context(self, available_actions: Dict[str, ActionInfo]) -> PreparedPlannerExecution:
        chat_context = collect_planner_chat_context(self.chat_id, self.last_obs_time_mark)
        self.last_obs_time_mark = chat_context.observed_at

        is_group_chat, chat_target_info, _ = self.get_necessary_info()
        filtered_actions = filter_actions_by_activation_type(
            available_actions,
            chat_context.chat_content_block_short,
            log_prefix=self.log_prefix,
        )
        logger.debug(f"{self.log_prefix}过滤后有{len(filtered_actions)}个可用动作")

        prompt_blocks, prompt, message_id_list = await self.build_planner_prompt(
            is_group_chat=is_group_chat,
            chat_target_info=chat_target_info,
            current_available_actions=filtered_actions,
            chat_content_block=chat_context.chat_content_block,
            message_id_list=chat_context.message_id_list,
            interest=global_config.personality.interest,
        )
        return PreparedPlannerExecution(
            prompt=prompt,
            message_id_list=message_id_list,
            filtered_actions=filtered_actions,
            prompt_blocks=prompt_blocks,
        )

    async def execute_prepared_plan(
        self,
        *,
        prepared_plan: PreparedPlannerExecution,
        available_actions: Dict[str, ActionInfo],
        loop_start_time: float,
    ) -> List[ActionPlannerInfo]:
        return await self._execute_main_planner(
            prompt=prepared_plan.prompt,
            message_id_list=prepared_plan.message_id_list,
            filtered_actions=prepared_plan.filtered_actions,
            available_actions=available_actions,
            loop_start_time=loop_start_time,
            prompt_blocks=prepared_plan.prompt_blocks,
        )

    async def build_planner_prompt(
        self,
        is_group_chat: bool,
        chat_target_info: Optional["TargetPersonInfo"],
        current_available_actions: Dict[str, ActionInfo],
        message_id_list: List[Tuple[str, "DatabaseMessages"]],
        chat_content_block: str = "",
        interest: str = "",
    ) -> tuple[List[PromptBlock], str, List[Tuple[str, "DatabaseMessages"]]]:
        del is_group_chat, current_available_actions

        try:
            actions_before_now = get_actions_by_timestamp_with_chat(
                chat_id=self.chat_id,
                timestamp_start=time.time() - 600,
                timestamp_end=time.time(),
                limit=6,
            )
            actions_before_now_block = build_readable_actions(actions=actions_before_now)
            if actions_before_now_block:
                actions_before_now_block = f"你刚刚选择并执行过的action是：\n{actions_before_now_block}"
            else:
                actions_before_now_block = ""

            chat_context_description = "你正在和对方聊天中"
            if chat_target_info:
                chat_context_description = (
                    f"你正在和 {chat_target_info.person_name or chat_target_info.user_nickname or '对方'} 聊天中"
                )

            time_block = f"当前时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            assembled = ContextAssembler().from_parts(
                system_parts={
                    "policy": _build_brain_planner_policy_block(),
                    "identity": _build_brain_planner_identity_block(interest),
                },
                user_parts={
                    "state": _build_brain_planner_state_block(
                        time_block=time_block,
                        chat_context_description=chat_context_description,
                        actions_before_now_block=actions_before_now_block,
                    ),
                    "chat_history": chat_content_block,
                    "task": _build_brain_planner_task_block(),
                },
            )
            planner_profile = build_agent_profile("planner_default")
            prepared_prompt = PromptComposer().prepare(assembled.blocks, planner_profile)
            composed = PromptComposer().compose(prepared_prompt.blocks)
            return prepared_prompt.blocks, composed.user_prompt, message_id_list
        except Exception as exc:
            logger.error(f"构建 Planner 提示词时出错: {exc}")
            logger.error(traceback.format_exc())
            return [], "构建 Planner Prompt 时出错", []

    def get_necessary_info(self) -> Tuple[bool, Optional["TargetPersonInfo"], Dict[str, ActionInfo]]:
        return resolve_necessary_info(
            self.chat_id,
            self.action_manager,
            log_prefix=self.log_prefix,
        )

    async def _execute_main_planner(
        self,
        prompt: str,
        message_id_list: List[Tuple[str, "DatabaseMessages"]],
        filtered_actions: Dict[str, ActionInfo],
        available_actions: Dict[str, ActionInfo],
        loop_start_time: float,
        prompt_blocks: Optional[List[PromptBlock]] = None,
    ) -> List[ActionPlannerInfo]:
        _reasoning, actions = await execute_planner_llm(
            self.planner_llm,
            PreparedPlannerExecution(
                prompt=prompt,
                message_id_list=message_id_list,
                filtered_actions=filtered_actions,
                prompt_blocks=prompt_blocks or [],
            ),
            available_actions=available_actions,
            loop_start_time=loop_start_time,
            log_prefix=self.log_prefix,
        )
        return actions


init_prompt()
