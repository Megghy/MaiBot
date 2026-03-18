from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from src.chat.planner_actions.planner_tools import (
    convert_actions_to_tools,
    create_no_reply,
    parse_single_tool_call,
    try_parse_json_fallback,
    try_parse_tagged_tool_fallback,
)
from src.chat.utils.chat_message_builder import (
    build_readable_messages_with_id,
    get_raw_msg_before_timestamp_with_chat,
)
from src.common.data_models.info_data_model import ActionPlannerInfo
from src.common.logger import get_logger
from src.config.config import global_config
from src.llm_models.utils_model import LLMRequest
from src.plugin_system.base.component_types import ActionInfo

if TYPE_CHECKING:
    from src.common.data_models.database_data_model import DatabaseMessages

logger = get_logger("planner")


@dataclass(slots=True)
class PlannerChatContext:
    chat_content_block: str
    chat_content_block_short: str
    message_id_list: List[Tuple[str, "DatabaseMessages"]]
    observed_at: float


@dataclass(slots=True)
class PreparedPlannerExecution:
    prompt: str
    message_id_list: List[Tuple[str, "DatabaseMessages"]]
    filtered_actions: Dict[str, ActionInfo]
    system_prompt: Optional[str] = None
    include_no_reply_until_call: bool = False
    include_reply: bool = True


def collect_planner_chat_context(chat_id: str, last_obs_time_mark: float) -> PlannerChatContext:
    message_list_before_now = get_raw_msg_before_timestamp_with_chat(
        chat_id=chat_id,
        timestamp=time.time(),
        limit=int(global_config.chat.max_context_size * 0.6),
    )
    chat_content_block, message_id_list = build_readable_messages_with_id(
        messages=message_list_before_now,
        timestamp_mode="normal_no_YMD",
        read_mark=last_obs_time_mark,
        truncate=True,
        show_actions=True,
        show_pic_mapping_header=False,
    )

    message_list_before_now_short = message_list_before_now[-int(global_config.chat.max_context_size * 0.3) :]
    chat_content_block_short, _ = build_readable_messages_with_id(
        messages=message_list_before_now_short,
        timestamp_mode="normal_no_YMD",
        truncate=False,
        show_actions=False,
        show_pic_mapping_header=False,
    )

    return PlannerChatContext(
        chat_content_block=chat_content_block,
        chat_content_block_short=chat_content_block_short,
        message_id_list=message_id_list,
        observed_at=time.time(),
    )


async def execute_planner_llm(
    planner_llm: LLMRequest,
    prepared_execution: PreparedPlannerExecution,
    *,
    available_actions: Dict[str, ActionInfo],
    loop_start_time: float,
    log_prefix: str = "",
) -> Tuple[str, List[ActionPlannerInfo]]:
    llm_content = ""
    reasoning_text = ""
    tool_calls = None
    actions: List[ActionPlannerInfo] = []

    try:
        tools = convert_actions_to_tools(
            prepared_execution.filtered_actions,
            include_no_reply_until_call=prepared_execution.include_no_reply_until_call,
            include_reply=prepared_execution.include_reply,
        )
        llm_content, (reasoning_content, _, tool_calls) = await planner_llm.generate_response_async(
            prompt=prepared_execution.prompt,
            tools=tools,
            system_prompt=prepared_execution.system_prompt,
        )
        reasoning_text = reasoning_content or llm_content or ""

        if global_config.debug.show_planner_prompt:
            logger.info(f"{log_prefix}规划器原始提示词: {prepared_execution.prompt}")
            if prepared_execution.system_prompt:
                logger.info(f"{log_prefix}规划器系统提示词: {prepared_execution.system_prompt}")
            logger.info(f"{log_prefix}规划器原始响应: {llm_content}")
            if reasoning_content:
                logger.info(f"{log_prefix}规划器推理: {reasoning_content}")
            if tool_calls:
                logger.info(f"{log_prefix}规划器工具调用: {[tool.func_name for tool in tool_calls]}")
        else:
            logger.debug(f"{log_prefix}规划器原始提示词: {prepared_execution.prompt}")
            if prepared_execution.system_prompt:
                logger.debug(f"{log_prefix}规划器系统提示词: {prepared_execution.system_prompt}")
            logger.debug(f"{log_prefix}规划器原始响应: {llm_content}")
            if reasoning_content:
                logger.debug(f"{log_prefix}规划器推理: {reasoning_content}")
    except Exception as exc:
        logger.error(f"{log_prefix}LLM 请求执行失败: {exc}")
        return f"LLM 请求失败，模型出现问题: {exc}", create_no_reply(
            f"LLM 请求失败，模型出现问题: {exc}",
            available_actions,
        )

    if tool_calls:
        try:
            logger.debug(f"{log_prefix}从响应中提取到{len(tool_calls)}个工具调用")
            filtered_actions_list = list(prepared_execution.filtered_actions.items())
            for tool_call in tool_calls:
                actions.extend(
                    parse_single_tool_call(
                        tool_call,
                        prepared_execution.message_id_list,
                        filtered_actions_list,
                        log_prefix=log_prefix,
                        extracted_reasoning=reasoning_text,
                    )
                )
        except Exception as exc:
            logger.warning(f"{log_prefix}解析LLM工具调用失败 {exc}")
            actions = create_no_reply(f"解析LLM工具调用失败: {exc}", available_actions)
    else:
        try:
            fallback_actions = try_parse_json_fallback(
                llm_content,
                prepared_execution.filtered_actions,
                prepared_execution.message_id_list,
                reasoning_text,
                log_prefix=log_prefix,
            )
            if not fallback_actions:
                fallback_actions = try_parse_tagged_tool_fallback(
                    llm_content,
                    prepared_execution.filtered_actions,
                    prepared_execution.message_id_list,
                    reasoning_text,
                    log_prefix=log_prefix,
                )
            if fallback_actions:
                actions.extend(fallback_actions)
            else:
                logger.warning(f"{log_prefix}LLM没有返回工具调用: {llm_content}")
                reason = (
                    "规划器未选择动作（LLM 返回的内容未能解析为有效的工具调用）"
                    if llm_content
                    else "规划器没有返回有效的动作选择"
                )
                actions = create_no_reply(reason, available_actions)
        except Exception as exc:
            logger.warning(f"{log_prefix}尝试解析内容为 JSON 失败: {exc}")
            reason = (
                "规划器未选择动作（LLM 返回的内容未能解析为有效的工具调用）"
                if llm_content
                else "规划器没有返回有效的动作选择"
            )
            actions = create_no_reply(reason, available_actions)

    for action in actions:
        action.action_data = action.action_data or {}
        action.action_data["loop_start_time"] = loop_start_time

    logger.debug(f"{log_prefix}规划器选择了{len(actions)}个动作: {' '.join([a.action_type for a in actions])}")
    return reasoning_text, actions
