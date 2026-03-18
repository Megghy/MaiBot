from __future__ import annotations

import time
from typing import Tuple

from src.chat.knowledge import get_qa_manager
from src.chat.utils.prompt_builder import global_prompt_manager
from src.common.logger import get_logger
from src.config.config import global_config, model_config
from src.llm_models.payload_content.message import Message
from src.llm_models.payload_content.tool_option import ToolParamType
from src.memory_system.retrieval_agent_helpers import (
    TerminalToolOutcome,
    ToolExecutionOutcome,
    _append_reasoning_summary,
    _append_tool_observation_message,
    _build_assistant_message,
    _build_message_factory,
    _execute_tool_calls,
    _resolve_terminal_tool_calls,
)
from src.memory_system.retrieval_format import (
    _append_collected_info,
    _calculate_iteration_budget,
    _has_successful_observation,
    _prepare_collected_info_for_prompt,
)
from src.memory_system.retrieval_tools import get_tool_registry
from src.plugin_system.apis import llm_api

logger = get_logger("memory_retrieval")

SPECIAL_TOOL_DEFINITIONS = [
    {
        "name": "found_answer",
        "description": "当找到足够的证据或根据常识可以回答问题时，调用此工具提交最终答案。调用此工具将结束思考过程。",
        "parameters": [
            (
                "answer",
                ToolParamType.STRING,
                "最终的答案内容。如果是基于检索到的信息，请综合整理；如果是基于常识，请直接给出。",
                True,
                None,
            ),
            (
                "source",
                ToolParamType.STRING,
                "答案的来源类型",
                True,
                ["memory", "model"],
            ),
        ],
    },
    {
        "name": "not_enough_info",
        "description": "当尝试了多种检索方式仍无法找到足够信息，且无法根据常识回答时，调用此工具结束思考过程。",
        "parameters": [
            (
                "reason",
                ToolParamType.STRING,
                "无法回答的具体原因",
                True,
                None,
            ),
        ],
    },
]


def _build_lpmm_hint() -> str:
    try:
        qa_manager = get_qa_manager()
        if qa_manager and qa_manager.embed_manager and qa_manager.embed_manager.paragraphs_embedding_store.store:
            return "- 如果不确定查询类别, 也可以使用 lpmm 知识库查询"
    except Exception as exc:
        logger.warning(f"检查LPMM内容失败: {exc}")
    return ""


async def _build_iteration_prompt(
    question: str,
    collected_info: str,
    *,
    iteration: int,
    max_iterations: int,
    tool_registry,
) -> tuple[str, list[dict[str, object]], bool]:
    bot_name = global_config.bot.nickname
    time_now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    current_iteration = iteration + 1
    remaining_iterations = max_iterations - current_iteration
    is_final_iteration = current_iteration >= max_iterations

    if is_final_iteration:
        tool_definitions = list(SPECIAL_TOOL_DEFINITIONS)
        logger.info(
            f"ReAct Agent 第 {current_iteration} 次迭代，问题: {question}|可用工具数量: {len(tool_definitions)}（最后一次迭代，仅提供提交工具）"
        )
        prompt = await global_prompt_manager.format_prompt(
            "memory_retrieval_react_final_prompt",
            bot_name=bot_name,
            time_now=time_now,
            question=question,
            collected_info=_prepare_collected_info_for_prompt(collected_info),
            current_iteration=current_iteration,
            remaining_iterations=remaining_iterations,
            max_iterations=max_iterations,
        )
        return prompt, tool_definitions, True

    tool_definitions = tool_registry.get_tool_definitions()
    tool_definitions.extend(SPECIAL_TOOL_DEFINITIONS)
    logger.info(f"ReAct Agent 第 {current_iteration} 次迭代，问题: {question}|可用工具数量: {len(tool_definitions)}")
    prompt = await global_prompt_manager.format_prompt(
        "memory_retrieval_react_prompt_head",
        bot_name=bot_name,
        time_now=time_now,
        question=question,
        collected_info=_prepare_collected_info_for_prompt(collected_info),
        current_iteration=current_iteration,
        remaining_iterations=remaining_iterations,
        max_iterations=max_iterations,
        lpmm_hint=_build_lpmm_hint(),
    )
    return prompt, tool_definitions, False


async def _react_agent_solve_question(
    question: str,
    chat_id: str,
    max_iterations: int = 5,
    timeout: float = 30.0,
    initial_info: str = "",
) -> Tuple[bool, str, list[dict[str, object]], bool]:
    """使用ReAct架构的Agent来解决问题"""
    start_time = time.time()
    collected_info = _append_collected_info("", initial_info)
    configured_budget = max(1, max_iterations or 1)
    dynamic_budget = _calculate_iteration_budget(question, collected_info)
    max_iterations = max(1, min(configured_budget, dynamic_budget or configured_budget))
    thinking_steps: list[dict[str, object]] = []
    is_timeout = False
    conversation_messages: list[Message] = []

    for iteration in range(max_iterations):
        if time.time() - start_time > timeout:
            logger.warning(f"ReAct Agent超时，已迭代{iteration}次")
            is_timeout = True
            break

        tool_registry = get_tool_registry()
        prompt, tool_definitions, is_final_iteration = await _build_iteration_prompt(
            question,
            collected_info,
            iteration=iteration,
            max_iterations=max_iterations,
            tool_registry=tool_registry,
        )

        (
            success,
            response,
            reasoning_content,
            model_name,
            tool_calls,
        ) = await llm_api.generate_with_model_with_tools_by_message_factory(
            _build_message_factory(
                prompt,
                conversation_messages,
                show_memory_prompt=global_config.debug.show_memory_prompt,
            ),
            model_config=model_config.model_task_config.tool_use,
            tool_options=tool_definitions,
            request_type="memory.react",
        )

        logger.info(
            f"ReAct Agent 第 {iteration + 1} 次迭代 模型: {model_name} ，调用工具数量: {len(tool_calls) if tool_calls else 0} ，调用工具响应: {response}"
        )

        if not success:
            logger.error(f"ReAct Agent LLM调用失败: {response}")
            break

        assistant_message = _build_assistant_message(response, tool_calls or [])
        step: dict[str, object] = {"iteration": iteration + 1, "thought": response, "actions": [], "observations": []}

        if assistant_message:
            conversation_messages.append(assistant_message)

        if reasoning_content or response:
            collected_info = _append_reasoning_summary(
                collected_info,
                reasoning_content,
                response,
                _append_collected_info,
            )

        if not tool_calls:
            if response and response.strip():
                step["observations"] = [f"思考完成，但未调用工具。响应: {response}"]
                logger.info(f"ReAct Agent 第 {iteration + 1} 次迭代 思考完成但未调用工具: {response[:100]}...")
                collected_info = _append_collected_info(collected_info, f"思考: {response}")
                thinking_steps.append(step)
                continue

            logger.warning(f"ReAct Agent 第 {iteration + 1} 次迭代 无工具调用且无响应")
            step["observations"] = ["无响应且无工具调用"]
            thinking_steps.append(step)
            break

        terminal_outcome = _resolve_terminal_tool_calls(tool_calls, step, thinking_steps, iteration)
        if terminal_outcome is not None:
            return terminal_outcome.found_answer, terminal_outcome.payload, thinking_steps, False

        tool_outcomes = await _execute_tool_calls(
            tool_registry,
            tool_calls,
            chat_id=chat_id,
            iteration=iteration,
            is_final_iteration=is_final_iteration,
        )
        for outcome in tool_outcomes:
            step["actions"].append({"action_type": outcome.action_type, "action_params": outcome.action_params})
            step["observations"].append(outcome.observation)
            collected_info = _append_collected_info(collected_info, outcome.observation)
            _append_tool_observation_message(
                conversation_messages,
                tool_call_id=outcome.tool_call_id,
                observation_text=outcome.observation,
            )

        thinking_steps.append(step)

    if collected_info:
        logger.warning(
            f"ReAct Agent达到最大迭代次数或超时，但未明确返回found_answer。已收集信息: {collected_info[:100]}..."
        )

    has_success = _has_successful_observation(thinking_steps)
    if is_timeout:
        logger.warning("ReAct Agent超时，直接视为not_enough_info")
        fallback_answer = "查询超时，未能在限制时间内完成信息整理"
    else:
        logger.warning("ReAct Agent达到最大迭代次数，直接视为not_enough_info")
        if has_success:
            fallback_answer = "已检索到相关记录，但Agent未能在限定轮次内整理出明确答案"
        else:
            fallback_answer = "未找到相关信息"

    return False, fallback_answer, thinking_steps, is_timeout


__all__ = [
    "SPECIAL_TOOL_DEFINITIONS",
    "TerminalToolOutcome",
    "ToolExecutionOutcome",
    "_build_iteration_prompt",
    "_build_lpmm_hint",
    "_execute_tool_calls",
    "_react_agent_solve_question",
    "_resolve_terminal_tool_calls",
]
