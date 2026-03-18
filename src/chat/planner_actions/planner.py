import time
from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

from rich.traceback import install

from src.chat.message_receive.chat_stream import get_chat_manager
from src.chat.planner_actions.action_manager import ActionManager
from src.chat.planner_actions.planner_prompting import build_planner_prompts, init_prompt
from src.chat.planner_actions.planner_runtime import (
    PreparedPlannerExecution,
    collect_planner_chat_context,
    execute_planner_llm,
)
from src.chat.planner_actions.planner_selection import (
    filter_actions_by_activation_type,
    get_necessary_info as resolve_necessary_info,
    select_actions_for_prompt,
)
from src.common.data_models.info_data_model import ActionPlannerInfo
from src.common.logger import get_logger
from src.config.config import global_config, model_config
from src.llm_models.utils_model import LLMRequest
from src.plugin_system.base.component_types import ActionInfo

if TYPE_CHECKING:
    from src.common.data_models.database_data_model import DatabaseMessages
    from src.common.data_models.info_data_model import TargetPersonInfo

logger = get_logger("planner")

MAX_DYNAMIC_ACTION_TOOLS = 6
RECENT_REPLY_WINDOW_SECONDS = 600

install(extra_lines=3)


class ActionPlanner:
    def __init__(self, chat_id: str, action_manager: ActionManager):
        self.chat_id = chat_id
        self.log_prefix = f"[{get_chat_manager().get_stream_name(chat_id) or chat_id}]"
        self.action_manager = action_manager
        self.planner_llm = LLMRequest(model_set=model_config.model_task_config.planner, request_type="planner")
        self.last_obs_time_mark = 0.0
        self.plan_log: List[Tuple[str, float, Union[List[ActionPlannerInfo], str]]] = []
        self._allow_no_reply_until_call = False
        self._allow_reply_action = True

    async def plan(
        self,
        available_actions: Dict[str, ActionInfo],
        loop_start_time: float = 0.0,
        is_mentioned: bool = False,
    ) -> List[ActionPlannerInfo]:
        self._allow_no_reply_until_call = False

        chat_context = collect_planner_chat_context(self.chat_id, self.last_obs_time_mark)
        self.last_obs_time_mark = chat_context.observed_at

        is_group_chat, chat_target_info, current_available_actions = self.get_necessary_info()
        filtered_actions = filter_actions_by_activation_type(
            available_actions,
            chat_context.chat_content_block_short,
            log_prefix=self.log_prefix,
        )
        selected_actions = select_actions_for_prompt(
            filtered_actions,
            chat_context.chat_content_block_short,
            max_dynamic_action_tools=MAX_DYNAMIC_ACTION_TOOLS,
        )
        if not selected_actions and filtered_actions:
            selected_actions = filtered_actions

        logger.debug(f"{self.log_prefix}过滤后有{len(filtered_actions)}个可用动作")
        if is_mentioned and not filtered_actions:
            logger.info(f"{self.log_prefix}提及时没有可用动作，跳过plan调用")
            return []

        prompt, system_prompt, message_id_list = await self.build_planner_prompt(
            is_group_chat=is_group_chat,
            chat_target_info=chat_target_info,
            current_available_actions=selected_actions,
            chat_content_block=chat_context.chat_content_block,
            message_id_list=chat_context.message_id_list,
            interest=global_config.personality.interest,
            is_mentioned=is_mentioned,
        )

        reasoning, actions = await self._execute_main_planner(
            prepared_execution=PreparedPlannerExecution(
                prompt=prompt,
                message_id_list=message_id_list,
                filtered_actions=selected_actions,
                system_prompt=system_prompt,
                include_no_reply_until_call=self._allow_no_reply_until_call,
                include_reply=self._allow_reply_action,
            ),
            available_actions=available_actions,
            loop_start_time=loop_start_time,
        )

        if actions:
            action_reasons = []
            for action in actions:
                action_reason = action.reasoning or ""
                action_reasons.append(f"{action.action_type}:{action_reason or '（无reason）'}")
            log_reasoning = " | ".join(action_reasons)
        else:
            log_reasoning = reasoning or "（无reason）"

        logger.info(
            f"{self.log_prefix}Planner:{log_reasoning}。选择了{len(actions)}个动作: {' '.join([a.action_type for a in actions])}"
        )
        self.add_plan_log(reasoning, actions)
        return actions

    def add_plan_log(self, reasoning: str, actions: List[ActionPlannerInfo]):
        self.plan_log.append((reasoning, time.time(), actions))
        if len(self.plan_log) > 20:
            self.plan_log.pop(0)

    def add_plan_excute_log(self, result: str):
        self.plan_log.append(("", time.time(), result))
        if len(self.plan_log) > 20:
            self.plan_log.pop(0)

    def get_plan_log_str(self, max_action_records: int = 2, max_execution_records: int = 5) -> str:
        action_records = []
        execution_records = []

        for reasoning, timestamp, content in reversed(self.plan_log):
            if isinstance(content, list) and all(isinstance(action, ActionPlannerInfo) for action in content):
                if len(action_records) < max_action_records:
                    summaries = []
                    for action in content:
                        try:
                            reason_text = (action.reasoning or "").strip()
                            summaries.append(f"{action.action_type}:{reason_text or '（无reason）'}")
                        except Exception:
                            summaries.append(f"{getattr(action, 'action_type', 'unknown')}:（无reason）")
                    action_records.append((" | ".join(summaries), timestamp, content, "action"))
            elif len(execution_records) < max_execution_records:
                execution_records.append((reasoning, timestamp, content, "execution"))

        all_records = action_records + execution_records
        all_records.sort(key=lambda item: item[1])

        plan_log_str = ""
        for reasoning, timestamp, content, record_type in all_records:
            time_str = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")
            if record_type == "action":
                plan_log_str += f"{time_str}:{reasoning}\n"
            else:
                plan_log_str += f"{time_str}:你执行了action:{content}\n"

        return plan_log_str

    def _has_consecutive_no_reply(self, min_count: int = 3) -> bool:
        consecutive_count = 0
        for _reasoning, _timestamp, content in reversed(self.plan_log):
            if isinstance(content, list) and all(isinstance(action, ActionPlannerInfo) for action in content):
                if all(action.action_type == "no_reply" for action in content):
                    consecutive_count += 1
                    if consecutive_count >= min_count:
                        return True
                else:
                    break
        return False

    async def build_planner_prompt(
        self,
        is_group_chat: bool,
        chat_target_info: Optional["TargetPersonInfo"],
        current_available_actions: Dict[str, ActionInfo],
        message_id_list: List[Tuple[str, "DatabaseMessages"]],
        chat_content_block: str = "",
        interest: str = "",
        is_mentioned: bool = False,
    ) -> tuple[str, str, List[Tuple[str, "DatabaseMessages"]]]:
        del is_group_chat, chat_target_info, current_available_actions

        prompt, system_prompt, allow_no_reply_until_call, allow_reply_action = await build_planner_prompts(
            interest=interest,
            chat_content_block=chat_content_block,
            is_mentioned=is_mentioned,
            has_consecutive_no_reply=self._has_consecutive_no_reply(min_count=3),
            plan_log=self.plan_log,
            reply_window_seconds=RECENT_REPLY_WINDOW_SECONDS,
        )
        self._allow_no_reply_until_call = allow_no_reply_until_call
        self._allow_reply_action = allow_reply_action
        return prompt, system_prompt, message_id_list

    def get_necessary_info(self) -> Tuple[bool, Optional["TargetPersonInfo"], Dict[str, ActionInfo]]:
        return resolve_necessary_info(
            self.chat_id,
            self.action_manager,
            log_prefix=self.log_prefix,
        )

    async def _execute_main_planner(
        self,
        prepared_execution: PreparedPlannerExecution,
        available_actions: Dict[str, ActionInfo],
        loop_start_time: float,
    ) -> Tuple[str, List[ActionPlannerInfo]]:
        return await execute_planner_llm(
            self.planner_llm,
            prepared_execution,
            available_actions=available_actions,
            loop_start_time=loop_start_time,
            log_prefix=self.log_prefix,
        )


init_prompt()
