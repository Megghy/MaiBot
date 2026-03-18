from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from typing import Any, Optional

from src.common.logger import get_logger
from src.llm_models.payload_content.message import Message, MessageBuilder, RoleType
from src.llm_models.payload_content.tool_option import ToolCall

logger = get_logger("memory_retrieval")

MAX_OBSERVATION_LENGTH = 1000
TERMINAL_TOOL_NAMES = {"found_answer", "not_enough_info"}


@dataclass(slots=True)
class TerminalToolOutcome:
    found_answer: bool
    payload: str


@dataclass(slots=True)
class PlannedToolExecution:
    action_type: str
    action_params: dict[str, Any]
    tool_call_id: Optional[str]
    coroutine: Any = None
    immediate_observation: Optional[str] = None


@dataclass(slots=True)
class ToolExecutionOutcome:
    action_type: str
    action_params: dict[str, Any]
    observation: str
    tool_call_id: Optional[str] = None


def _log_messages(messages: list[Message], *, show_memory_prompt: bool = False) -> None:
    if not show_memory_prompt:
        return

    log_lines = []
    for idx, msg in enumerate(messages, 1):
        role_name = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
        if isinstance(msg.content, str):
            full_content = msg.content
            content_type = "文本"
        elif isinstance(msg.content, list):
            text_parts = [item for item in msg.content if isinstance(item, str)]
            image_count = len([item for item in msg.content if isinstance(item, tuple)])
            full_content = "".join(text_parts) if text_parts else ""
            content_type = f"混合({len(text_parts)}段文本, {image_count}张图片)"
        else:
            full_content = str(msg.content)
            content_type = "未知"

        msg_info = (
            f"\n[消息 {idx}] 角色: {role_name} 内容类型: {content_type}\n========================================"
        )
        if full_content:
            msg_info += f"\n{full_content}"
        if msg.tool_calls:
            msg_info += f"\n  工具调用: {len(msg.tool_calls)}个"
            for tool_call in msg.tool_calls:
                msg_info += f"\n    - {tool_call}"
        if msg.tool_call_id:
            msg_info += f"\n  工具调用ID: {msg.tool_call_id}"
        log_lines.append(msg_info)

    logger.info(f"消息列表 (共{len(messages)}条):{''.join(log_lines)}")


def _build_message_factory(prompt: str, conversation_messages: list[Message], *, show_memory_prompt: bool = False):
    def message_factory(
        _client,
        *,
        _prompt: str = prompt,
        _conversation_messages: list[Message] = conversation_messages,
    ) -> list[Message]:
        messages: list[Message] = []

        system_builder = MessageBuilder()
        system_builder.set_role(RoleType.System)
        system_builder.add_text_content(_prompt)
        messages.append(system_builder.build())
        messages.extend(_conversation_messages)

        _log_messages(messages, show_memory_prompt=show_memory_prompt)
        return messages

    return message_factory


def _build_assistant_message(response: str, tool_calls: list[ToolCall]) -> Optional[Message]:
    if tool_calls:
        assistant_builder = MessageBuilder()
        assistant_builder.set_role(RoleType.Assistant)
        if response and response.strip():
            assistant_builder.add_text_content(response)
        assistant_builder.set_tool_calls(tool_calls)
        return assistant_builder.build()

    if response and response.strip():
        assistant_builder = MessageBuilder()
        assistant_builder.set_role(RoleType.Assistant)
        assistant_builder.add_text_content(response)
        return assistant_builder.build()

    return None


def _append_reasoning_summary(collected_info: str, reasoning_content: str, response: str, append_collected_info) -> str:
    thought_summary = reasoning_content or (response[:200] if response else "")
    if not thought_summary:
        return collected_info
    return append_collected_info(collected_info, f"[思考] {thought_summary}")


def _resolve_terminal_tool_calls(
    tool_calls: list[ToolCall],
    step: dict[str, Any],
    thinking_steps: list[dict[str, Any]],
    iteration: int,
) -> Optional[TerminalToolOutcome]:
    for tool_call in tool_calls:
        tool_name = tool_call.func_name
        tool_args = tool_call.args or {}

        if tool_name == "found_answer":
            answer = tool_args.get("answer", "")
            step["actions"].append({"action_type": "found_answer", "action_params": tool_args})
            step["observations"] = ["Agent调用found_answer提交答案"]
            thinking_steps.append(step)
            logger.info(f"ReAct Agent 第 {iteration + 1} 次迭代 调用found_answer: {answer[:100]}...")
            return TerminalToolOutcome(found_answer=True, payload=answer)

        if tool_name == "not_enough_info":
            reason = tool_args.get("reason", "未知原因")
            step["actions"].append({"action_type": "not_enough_info", "action_params": tool_args})
            step["observations"] = ["Agent调用not_enough_info提交原因"]
            thinking_steps.append(step)
            logger.info(f"ReAct Agent 第 {iteration + 1} 次迭代 调用not_enough_info: {reason[:100]}...")
            return TerminalToolOutcome(found_answer=False, payload=reason)

    return None


def _prepare_tool_params(tool, tool_args: dict[str, Any], chat_id: str) -> dict[str, Any]:
    params = tool_args.copy()
    signature = inspect.signature(tool.execute_func)
    if "chat_id" in signature.parameters:
        params["chat_id"] = chat_id
    return params


async def _execute_single_tool(tool, params: dict[str, Any], tool_name: str, iteration: int) -> str:
    try:
        observation = await tool.execute(**params)
    except Exception as exc:
        error_msg = f"工具执行失败: {str(exc)}"
        logger.error(f"ReAct Agent 第 {iteration + 1} 次迭代 工具 {tool_name} {error_msg}")
        return f"查询{tool_name}失败: {error_msg}"

    param_str = ", ".join([f"{k}={v}" for k, v in params.items() if k != "chat_id"])
    return f"查询{tool_name}({param_str})的结果：{observation}"


def _truncate_observation(observation: Any) -> str:
    observation_text = observation if isinstance(observation, str) else str(observation)
    if len(observation_text) > MAX_OBSERVATION_LENGTH:
        return observation_text[:MAX_OBSERVATION_LENGTH] + "...(结果过长已截断)"
    return observation_text


async def _execute_tool_calls(
    tool_registry,
    tool_calls: list[ToolCall],
    *,
    chat_id: str,
    iteration: int,
    is_final_iteration: bool,
) -> list[ToolExecutionOutcome]:
    planned_calls: list[PlannedToolExecution] = []

    for index, tool_call in enumerate(tool_calls, 1):
        tool_name = tool_call.func_name
        if tool_name in TERMINAL_TOOL_NAMES:
            continue

        tool_args = tool_call.args or {}
        logger.info(
            f"ReAct Agent 第 {iteration + 1} 次迭代 工具调用 {index}/{len(tool_calls)}: {tool_name}({tool_args})"
        )

        if is_final_iteration:
            logger.warning(f"ReAct Agent 第 {iteration + 1} 次迭代 仍尝试调用检索工具 {tool_name}，按规则已忽略")
            planned_calls.append(
                PlannedToolExecution(
                    action_type=tool_name,
                    action_params=tool_args,
                    tool_call_id=None,
                    immediate_observation=f"最终迭代忽略检索工具调用: {tool_name}",
                )
            )
            continue

        tool = tool_registry.get_tool(tool_name)
        if not tool:
            error_msg = f"未知的工具类型: {tool_name}"
            logger.warning(f"ReAct Agent 第 {iteration + 1} 次迭代 工具 {index}/{len(tool_calls)} {error_msg}")
            planned_calls.append(
                PlannedToolExecution(
                    action_type=tool_name,
                    action_params=tool_args,
                    tool_call_id=None,
                    immediate_observation=f"查询{tool_name}失败: {error_msg}",
                )
            )
            continue

        tool_params = _prepare_tool_params(tool, tool_args, chat_id)
        planned_calls.append(
            PlannedToolExecution(
                action_type=tool_name,
                action_params=tool_args,
                tool_call_id=tool_call.call_id,
                coroutine=_execute_single_tool(tool, tool_params, tool_name, iteration),
            )
        )

    pending_coroutines = [planned.coroutine for planned in planned_calls if planned.coroutine is not None]
    gathered_results = await asyncio.gather(*pending_coroutines, return_exceptions=True) if pending_coroutines else []
    gathered_iter = iter(gathered_results)

    outcomes: list[ToolExecutionOutcome] = []
    for planned in planned_calls:
        if planned.coroutine is None:
            observation_text = planned.immediate_observation or ""
        else:
            observation = next(gathered_iter)
            if isinstance(observation, Exception):
                observation_text = f"工具执行异常: {str(observation)}"
                logger.error(f"ReAct Agent 第 {iteration + 1} 次迭代 工具执行异常: {observation_text}")
            else:
                observation_text = _truncate_observation(observation)

        outcomes.append(
            ToolExecutionOutcome(
                action_type=planned.action_type,
                action_params=planned.action_params,
                observation=observation_text,
                tool_call_id=planned.tool_call_id,
            )
        )

    return outcomes


def _append_tool_observation_message(
    conversation_messages: list[Message],
    *,
    tool_call_id: Optional[str],
    observation_text: str,
) -> None:
    if not tool_call_id or not observation_text.strip():
        return

    tool_builder = MessageBuilder()
    tool_builder.set_role(RoleType.Tool)
    tool_builder.add_text_content(observation_text)
    tool_builder.add_tool_call(tool_call_id)
    conversation_messages.append(tool_builder.build())


__all__ = [
    "TERMINAL_TOOL_NAMES",
    "TerminalToolOutcome",
    "ToolExecutionOutcome",
    "_append_reasoning_summary",
    "_append_tool_observation_message",
    "_build_assistant_message",
    "_build_message_factory",
    "_execute_tool_calls",
    "_resolve_terminal_tool_calls",
]
