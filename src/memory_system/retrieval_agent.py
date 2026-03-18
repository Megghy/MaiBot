import asyncio
import inspect
import time
from typing import Any, Dict, List, Optional, Tuple

from src.chat.knowledge import get_qa_manager
from src.chat.utils.prompt_builder import global_prompt_manager
from src.common.logger import get_logger
from src.config.config import global_config, model_config
from src.llm_models.payload_content.message import MessageBuilder, Message, RoleType
from src.llm_models.payload_content.tool_option import ToolParamType
from src.memory_system.retrieval_format import (
    _append_collected_info,
    _calculate_iteration_budget,
    _has_successful_observation,
    _prepare_collected_info_for_prompt,
)
from src.memory_system.retrieval_tools import get_tool_registry
from src.plugin_system.apis import llm_api

logger = get_logger("memory_retrieval")

MAX_OBSERVATION_LENGTH = 1000

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


async def _react_agent_solve_question(
    question: str,
    chat_id: str,
    max_iterations: int = 5,
    timeout: float = 30.0,
    initial_info: str = "",
) -> Tuple[bool, str, List[Dict[str, Any]], bool]:
    """使用ReAct架构的Agent来解决问题"""
    start_time = time.time()
    collected_info = _append_collected_info("", initial_info)
    configured_budget = max(1, max_iterations or 1)
    dynamic_budget = _calculate_iteration_budget(question, collected_info)
    max_iterations = max(1, min(configured_budget, dynamic_budget or configured_budget))
    thinking_steps = []
    is_timeout = False
    conversation_messages: List[Message] = []

    for iteration in range(max_iterations):
        if time.time() - start_time > timeout:
            logger.warning(f"ReAct Agent超时，已迭代{iteration}次")
            is_timeout = True
            break

        tool_registry = get_tool_registry()
        bot_name = global_config.bot.nickname
        time_now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        current_iteration = iteration + 1
        remaining_iterations = max_iterations - current_iteration
        is_final_iteration = current_iteration >= max_iterations

        if is_final_iteration:
            tool_definitions = SPECIAL_TOOL_DEFINITIONS
            logger.info(
                f"ReAct Agent 第 {iteration + 1} 次迭代，问题: {question}|可用工具数量: {len(tool_definitions)}（最后一次迭代，仅提供提交工具）"
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
        else:
            tool_definitions = tool_registry.get_tool_definitions()
            tool_definitions.extend(SPECIAL_TOOL_DEFINITIONS)
            logger.info(
                f"ReAct Agent 第 {iteration + 1} 次迭代，问题: {question}|可用工具数量: {len(tool_definitions)}"
            )

            lpmm_hint = ""
            try:
                qa_manager = get_qa_manager()
                if (
                    qa_manager
                    and qa_manager.embed_manager
                    and qa_manager.embed_manager.paragraphs_embedding_store.store
                ):
                    lpmm_hint = "- 如果不确定查询类别, 也可以使用 lpmm 知识库查询"
            except Exception as exc:
                logger.warning(f"检查LPMM内容失败: {exc}")

            prompt = await global_prompt_manager.format_prompt(
                "memory_retrieval_react_prompt_head",
                bot_name=bot_name,
                time_now=time_now,
                question=question,
                collected_info=_prepare_collected_info_for_prompt(collected_info),
                current_iteration=current_iteration,
                remaining_iterations=remaining_iterations,
                max_iterations=max_iterations,
                lpmm_hint=lpmm_hint,
            )

        def message_factory(
            _client,
            *,
            _prompt: str = prompt,
            _conversation_messages: List[Message] = conversation_messages,
        ) -> List[Message]:
            messages: List[Message] = []

            system_builder = MessageBuilder()
            system_builder.set_role(RoleType.System)
            system_builder.add_text_content(_prompt)
            messages.append(system_builder.build())
            messages.extend(_conversation_messages)

            if global_config.debug.show_memory_prompt:
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

                    msg_info = f"\n[消息 {idx}] 角色: {role_name} 内容类型: {content_type}\n========================================"
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

            return messages

        (
            success,
            response,
            reasoning_content,
            model_name,
            tool_calls,
        ) = await llm_api.generate_with_model_with_tools_by_message_factory(
            message_factory,
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

        assistant_message: Optional[Message] = None
        if tool_calls:
            assistant_builder = MessageBuilder()
            assistant_builder.set_role(RoleType.Assistant)
            if response and response.strip():
                assistant_builder.add_text_content(response)
            assistant_builder.set_tool_calls(tool_calls)
            assistant_message = assistant_builder.build()
        elif response and response.strip():
            assistant_builder = MessageBuilder()
            assistant_builder.set_role(RoleType.Assistant)
            assistant_builder.add_text_content(response)
            assistant_message = assistant_builder.build()

        step = {"iteration": iteration + 1, "thought": response, "actions": [], "observations": []}

        if assistant_message:
            conversation_messages.append(assistant_message)

        if reasoning_content or response:
            thought_summary = reasoning_content or (response[:200] if response else "")
            if thought_summary:
                collected_info = _append_collected_info(collected_info, f"[思考] {thought_summary}")

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

        tool_tasks = []

        for tool_call in tool_calls:
            tool_name = tool_call.func_name
            tool_args = tool_call.args or {}

            if tool_name == "found_answer":
                found_answer_content = tool_args.get("answer", "")
                step["actions"].append({"action_type": "found_answer", "action_params": tool_args})
                step["observations"] = ["Agent调用found_answer提交答案"]
                thinking_steps.append(step)
                logger.info(f"ReAct Agent 第 {iteration + 1} 次迭代 调用found_answer: {found_answer_content[:100]}...")
                return True, found_answer_content, thinking_steps, False

            if tool_name == "not_enough_info":
                not_enough_info_reason = tool_args.get("reason", "未知原因")
                step["actions"].append({"action_type": "not_enough_info", "action_params": tool_args})
                step["observations"] = ["Agent调用not_enough_info提交原因"]
                thinking_steps.append(step)
                logger.info(
                    f"ReAct Agent 第 {iteration + 1} 次迭代 调用not_enough_info: {not_enough_info_reason[:100]}..."
                )
                return False, not_enough_info_reason, thinking_steps, False

        for i, tool_call in enumerate(tool_calls):
            tool_name = tool_call.func_name
            tool_args = tool_call.args or {}

            logger.info(
                f"ReAct Agent 第 {iteration + 1} 次迭代 工具调用 {i + 1}/{len(tool_calls)}: {tool_name}({tool_args})"
            )

            if is_final_iteration and tool_name not in {"found_answer", "not_enough_info"}:
                logger.warning(f"ReAct Agent 第 {iteration + 1} 次迭代 仍尝试调用检索工具 {tool_name}，按规则已忽略")
                step["actions"].append({"action_type": tool_name, "action_params": tool_args})
                step["observations"].append(f"最终迭代忽略检索工具调用: {tool_name}")
                continue

            tool = tool_registry.get_tool(tool_name)
            if tool:
                tool_params = tool_args.copy()
                sig = inspect.signature(tool.execute_func)
                if "chat_id" in sig.parameters:
                    tool_params["chat_id"] = chat_id

                async def execute_single_tool(tool_instance, params, tool_name_str, iter_num):
                    try:
                        observation = await tool_instance.execute(**params)
                        param_str = ", ".join([f"{k}={v}" for k, v in params.items() if k != "chat_id"])
                        return f"查询{tool_name_str}({param_str})的结果：{observation}"
                    except Exception as exc:
                        error_msg = f"工具执行失败: {str(exc)}"
                        logger.error(f"ReAct Agent 第 {iter_num + 1} 次迭代 工具 {tool_name_str} {error_msg}")
                        return f"查询{tool_name_str}失败: {error_msg}"

                tool_tasks.append(execute_single_tool(tool, tool_params, tool_name, iteration))
                step["actions"].append({"action_type": tool_name, "action_params": tool_args})
            else:
                error_msg = f"未知的工具类型: {tool_name}"
                logger.warning(f"ReAct Agent 第 {iteration + 1} 次迭代 工具 {i + 1}/{len(tool_calls)} {error_msg}")
                tool_tasks.append(asyncio.create_task(asyncio.sleep(0, result=f"查询{tool_name}失败: {error_msg}")))

        if tool_tasks:
            observations = await asyncio.gather(*tool_tasks, return_exceptions=True)
            for i, (tool_call_item, observation) in enumerate(zip(tool_calls, observations, strict=False)):
                if isinstance(observation, Exception):
                    observation = f"工具执行异常: {str(observation)}"
                    logger.error(f"ReAct Agent 第 {iteration + 1} 次迭代 工具 {i + 1} 执行异常: {observation}")

                observation_text = observation if isinstance(observation, str) else str(observation)
                if len(observation_text) > MAX_OBSERVATION_LENGTH:
                    observation_text = observation_text[:MAX_OBSERVATION_LENGTH] + "...(结果过长已截断)"

                step["observations"].append(observation_text)
                collected_info = _append_collected_info(collected_info, observation_text)
                if observation_text.strip():
                    tool_builder = MessageBuilder()
                    tool_builder.set_role(RoleType.Tool)
                    tool_builder.add_text_content(observation_text)
                    tool_builder.add_tool_call(tool_call_item.call_id)
                    conversation_messages.append(tool_builder.build())

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
