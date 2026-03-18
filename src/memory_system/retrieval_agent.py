from __future__ import annotations

import time
from typing import Any, Tuple

from src.chat.knowledge import get_qa_manager
from src.chat.utils.prompt_builder import global_prompt_manager
from src.common.logger import get_logger
from src.config.config import global_config
from src.llm_models.payload_content.tool_option import ToolParamType
from src.memory_system.retrieval_agent_helpers import _append_reasoning_summary
from src.memory_system.retrieval_format import (
    _append_collected_info,
    _calculate_iteration_budget,
    _has_successful_observation,
    _prepare_collected_info_for_prompt,
)
from src.memory_system.retrieval_tools import build_agent_tool_registry
from src.runtime.agent import AgentRequest, AgentRunner, AgentStatus, PromptBlock, PromptBlockRole, build_agent_profile

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
    configured_budget = max(1, max_iterations or 1)
    collected_info = _append_collected_info("", initial_info)
    dynamic_budget = _calculate_iteration_budget(question, collected_info)
    max_iterations = max(1, min(configured_budget, dynamic_budget or configured_budget))

    agent_tool_registry = build_agent_tool_registry()

    loop_state: dict[str, Any] = {
        "question": question,
        "collected_info": collected_info,
        "legacy_registry": agent_tool_registry,
    }

    async def loop_factory(*, iteration: int, max_iterations: int, result, state, default_tool_definitions):
        updated_info = state.get("collected_info", "")
        last_reasoning = state.pop("last_reasoning", "")
        last_response = state.pop("last_response", "")
        if last_reasoning or last_response:
            updated_info = _append_reasoning_summary(
                updated_info,
                last_reasoning,
                last_response,
                _append_collected_info,
            )
        for observation in state.pop("pending_observations", []):
            updated_info = _append_collected_info(updated_info, observation)
        state["collected_info"] = updated_info

        prompt, tool_definitions, is_final_iteration = await _build_iteration_prompt(
            question,
            updated_info,
            iteration=iteration,
            max_iterations=max_iterations,
            tool_registry=state["legacy_registry"],
        )
        logger.info(
            f"ReAct Agent 第 {iteration + 1} 次迭代，问题: {question}|可用工具数量: {len(tool_definitions)}"
            + ("（最后一次迭代）" if is_final_iteration else "")
        )
        return {
            "prompt_blocks": [PromptBlock(name="task", content=prompt, role=PromptBlockRole.SYSTEM)],
            "tool_definitions": tool_definitions,
        }

    runner = AgentRunner(tool_registry=agent_tool_registry)
    profile = build_agent_profile(
        "memory_retrieval",
        max_iterations=max_iterations,
        timeout_seconds=timeout,
    )
    result = await runner.run(
        AgentRequest(
            request_id=f"memory-react:{chat_id}:{int(time.time() * 1000)}",
            chat_id=chat_id,
            user_id=None,
            message=question,
            skill_name="memory_retrieval",
            metadata={
                "tool_loop_factory": loop_factory,
                "tool_loop_state": loop_state,
            },
        ),
        profile,
    )

    thinking_steps = []
    for trace in result.trace:
        if trace.get("event") != "tool_loop_iteration":
            continue
        thinking_steps.append(
            {
                "iteration": trace.get("iteration"),
                "thought": trace.get("response", ""),
                "reasoning": trace.get("reasoning", ""),
                "actions": [
                    {
                        "action_type": action.get("tool_name", ""),
                        "action_params": action.get("args", {}) or {},
                    }
                    for action in trace.get("actions", [])
                ],
                "observations": list(trace.get("observations", [])),
            }
        )

    terminal = result.source_summary.get("terminal_tool")
    if terminal:
        tool_name = terminal.get("name", "")
        args = terminal.get("args", {}) or {}
        if tool_name == "found_answer":
            return True, str(args.get("answer", "")), thinking_steps, False
        if tool_name == "not_enough_info":
            return False, str(args.get("reason", "未知原因")), thinking_steps, False

    collected_info = str(loop_state.get("collected_info", "")).strip()
    if result.reasoning or result.output_text:
        collected_info = _append_reasoning_summary(
            collected_info,
            result.reasoning,
            result.output_text,
            _append_collected_info,
        )
    for step in thinking_steps:
        if step["actions"] or step["observations"]:
            continue
        thought = str(step.get("thought", "")).strip()
        if thought:
            step["observations"] = [f"思考完成，但未调用工具。响应: {thought}"]
    is_timeout = result.status == AgentStatus.TIMEOUT
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
    "_build_iteration_prompt",
    "_build_lpmm_hint",
    "_react_agent_solve_question",
]
