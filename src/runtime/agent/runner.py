from __future__ import annotations

import asyncio
import inspect
import uuid
from hashlib import sha1
from typing import Any

from src.common.logger import get_logger
from src.llm_models.payload_content.message import Message, MessageBuilder, RoleType
from src.llm_models.payload_content.tool_option import ToolCall
from src.runtime.agent.models import (
    AgentMode,
    AgentProfile,
    AgentRequest,
    AgentResult,
    AgentStatus,
    PromptBlock,
    PromptBlockRole,
    ToolCallRecord,
    ToolExecutionContext,
)
from src.runtime.agent.prompt_composer import PromptComposer
from src.runtime.agent.tool_registry import ToolRegistry, get_agent_tool_registry


logger = get_logger("agent_runner")


def _build_assistant_message(content: str, tool_calls: list[ToolCall]) -> Message | None:
    if not content.strip() and not tool_calls:
        return None
    builder = MessageBuilder().set_role(RoleType.Assistant)
    if content.strip():
        builder.add_text_content(content)
    if tool_calls:
        builder.set_tool_calls(tool_calls)
    return builder.build()


def _build_tool_message(tool_call_id: str, observation: str) -> Message | None:
    if not tool_call_id or not observation.strip():
        return None
    return MessageBuilder().set_role(RoleType.Tool).add_text_content(observation).add_tool_call(tool_call_id).build()


class AgentRunner:
    def __init__(
        self,
        *,
        tool_registry: ToolRegistry | None = None,
        prompt_composer: PromptComposer | None = None,
    ) -> None:
        self.tool_registry = tool_registry or get_agent_tool_registry()
        self.prompt_composer = prompt_composer or PromptComposer()

    async def run(
        self,
        request: AgentRequest,
        profile: AgentProfile,
        *,
        llm_request: Any | None = None,
        chat_stream: Any | None = None,
        session: Any | None = None,
        plugin_config: dict[str, Any] | None = None,
    ) -> AgentResult:
        blocks = request.prompt_blocks or [PromptBlock(name="context", content=request.message)]
        prepared_blocks = self.prompt_composer.prepare(blocks, profile)
        composed = self.prompt_composer.compose(prepared_blocks.blocks)
        result = AgentResult(
            status=AgentStatus.NO_RESULT,
            used_skill=profile.skill_name,
            system_prompt=composed.system_prompt,
            user_prompt=composed.user_prompt,
            trace=[
                {
                    "event": "compose_prompt",
                    "system_blocks": len(
                        [block for block in composed.ordered_blocks if block.role == PromptBlockRole.SYSTEM]
                    ),
                    "user_blocks": len(
                        [block for block in composed.ordered_blocks if block.role == PromptBlockRole.USER]
                    ),
                    "block_names": [block.name for block in composed.ordered_blocks],
                    "block_char_counts": {block.name: len(block.content) for block in composed.ordered_blocks},
                    "dropped_blocks": list(prepared_blocks.dropped_blocks),
                    "injected_blocks": list(prepared_blocks.injected_blocks),
                    "prompt_hash": sha1(
                        f"{composed.system_prompt}\n---\n{composed.user_prompt}".encode("utf-8")
                    ).hexdigest()[:12],
                }
            ],
        )
        if "reasoning_config" in request.metadata:
            result.source_summary["reasoning_config"] = request.metadata["reasoning_config"]

        llm = llm_request or self._build_llm_request(profile)
        context = ToolExecutionContext(
            chat_id=request.chat_id,
            chat_stream=chat_stream,
            session=session,
            plugin_config=dict(plugin_config or {}),
            metadata=dict(request.metadata or {}),
        )

        try:
            if profile.agent_mode == AgentMode.TOOL_LOOP:
                return await asyncio.wait_for(
                    self._run_tool_loop(result, request, profile, llm, context),
                    timeout=profile.timeout_seconds,
                )
            return await asyncio.wait_for(
                self._run_single_pass(
                    result,
                    profile,
                    llm,
                    tool_definitions_override=context.metadata.get("tool_definitions"),
                ),
                timeout=profile.timeout_seconds,
            )
        except asyncio.TimeoutError:
            result.status = AgentStatus.TIMEOUT
            result.error_message = f"Agent execution timed out after {profile.timeout_seconds}s"
            result.trace.append({"event": "timeout", "timeout_seconds": profile.timeout_seconds})
            return result
        except Exception as exc:
            logger.error(f"Agent runner 执行失败: {exc}")
            result.status = AgentStatus.ERROR
            result.error_message = str(exc)
            result.trace.append({"event": "error", "message": str(exc)})
            return result

    async def _run_single_pass(
        self,
        result: AgentResult,
        profile: AgentProfile,
        llm: Any,
        *,
        tool_definitions_override: list[dict[str, Any]] | None = None,
    ) -> AgentResult:
        tool_definitions = tool_definitions_override
        if tool_definitions is None:
            tool_definitions = self.tool_registry.get_tool_definitions(profile)
        llm_kwargs = self._build_single_pass_kwargs(
            llm,
            prompt=result.user_prompt,
            system_prompt=result.system_prompt or None,
            tools=tool_definitions or None,
            reasoning_config=result.source_summary.get("reasoning_config"),
        )
        response, (reasoning, model_name, tool_calls) = await llm.generate_response_async(**llm_kwargs)
        result.status = AgentStatus.SUCCESS if response or tool_calls else AgentStatus.NO_RESULT
        result.output_text = response or ""
        result.reasoning = reasoning or ""
        result.model_name = model_name or ""
        result.tool_calls = [
            ToolCallRecord(
                call_id=call.call_id,
                tool_name=call.func_name,
                args=dict(call.args or {}),
                executed=False,
            )
            for call in (tool_calls or [])
        ]
        result.used_tools = [record.tool_name for record in result.tool_calls]
        result.trace.append(
            {
                "event": "single_pass",
                "model_name": result.model_name,
                "tool_call_count": len(result.tool_calls),
            }
        )
        return result

    async def _run_tool_loop(
        self,
        result: AgentResult,
        request: AgentRequest,
        profile: AgentProfile,
        llm: Any,
        context: ToolExecutionContext,
    ) -> AgentResult:
        base_tool_definitions = self.tool_registry.get_tool_definitions(profile)
        conversation_messages: list[Message] = []
        observations: list[str] = []
        terminal_tool_name: str | None = None
        terminal_tool_args: dict[str, Any] | None = None
        loop_factory = context.metadata.get("tool_loop_factory")
        loop_state = context.metadata.setdefault("tool_loop_state", {})

        for iteration in range(max(1, profile.max_iterations)):
            system_prompt = result.system_prompt
            user_prompt = result.user_prompt
            tool_definitions = base_tool_definitions

            if loop_factory:
                factory_result = loop_factory(
                    iteration=iteration,
                    max_iterations=max(1, profile.max_iterations),
                    result=result,
                    state=loop_state,
                    default_tool_definitions=base_tool_definitions,
                )
                if inspect.isawaitable(factory_result):
                    factory_result = await factory_result
                if isinstance(factory_result, dict):
                    prompt_blocks = factory_result.get("prompt_blocks") or request.prompt_blocks
                    if prompt_blocks:
                        prepared_blocks = self.prompt_composer.prepare(prompt_blocks, profile)
                        composed = self.prompt_composer.compose(prepared_blocks.blocks)
                        system_prompt = composed.system_prompt
                        user_prompt = composed.user_prompt
                        loop_state["last_prompt_blocks"] = [block.name for block in composed.ordered_blocks]
                        loop_state["last_dropped_blocks"] = list(prepared_blocks.dropped_blocks)
                        loop_state["last_injected_blocks"] = list(prepared_blocks.injected_blocks)
                    tool_definitions = factory_result.get("tool_definitions") or base_tool_definitions
                result.system_prompt = system_prompt
                result.user_prompt = user_prompt

            response, (reasoning, model_name, tool_calls) = await llm.generate_response_with_message_async(
                message_factory=lambda _client, _system_prompt=system_prompt, _user_prompt=user_prompt: self._build_message_list(
                    _system_prompt,
                    _user_prompt,
                    conversation_messages,
                ),
                tools=tool_definitions or None,
            )
            result.model_name = model_name or result.model_name
            result.reasoning = reasoning or result.reasoning

            current_tool_calls = list(tool_calls or [])
            assistant_message = _build_assistant_message(response or "", current_tool_calls)
            if assistant_message is not None:
                conversation_messages.append(assistant_message)

            iteration_record = {
                "event": "tool_loop_iteration",
                "iteration": iteration + 1,
                "tool_call_count": len(current_tool_calls),
                "response": response or "",
                "reasoning": reasoning or "",
                "actions": [],
                "observations": [],
                "block_names": list(loop_state.get("last_prompt_blocks", [])),
                "dropped_blocks": list(loop_state.get("last_dropped_blocks", [])),
                "injected_blocks": list(loop_state.get("last_injected_blocks", [])),
            }
            result.trace.append(iteration_record)

            if not current_tool_calls:
                result.status = AgentStatus.SUCCESS if response else AgentStatus.NO_RESULT
                result.output_text = response or result.output_text
                result.observations = observations
                result.terminal_tool_name = terminal_tool_name
                loop_state["last_response"] = response or ""
                loop_state["last_reasoning"] = reasoning or ""
                loop_state["pending_observations"] = []
                loop_state["last_iteration"] = iteration + 1
                return result

            for call in current_tool_calls:
                record = ToolCallRecord(
                    call_id=call.call_id or f"tool-{uuid.uuid4().hex}",
                    tool_name=call.func_name,
                    args=dict(call.args or {}),
                )
                result.tool_calls.append(record)

                if call.func_name in profile.terminal_tools:
                    terminal_tool_name = call.func_name
                    terminal_tool_args = dict(call.args or {})
                    result.used_tools.append(record.tool_name)
                    iteration_record["actions"].append(
                        {
                            "tool_name": call.func_name,
                            "args": dict(call.args or {}),
                            "terminal": True,
                        }
                    )
                    if call.func_name == "found_answer":
                        iteration_record["observations"].append("Agent调用found_answer提交答案")
                    elif call.func_name == "not_enough_info":
                        iteration_record["observations"].append("Agent调用not_enough_info提交原因")
                    break

                spec = self.tool_registry.get(call.func_name)
                if spec is None:
                    missing_observation = f"工具{call.func_name}未注册"
                    observations.append(missing_observation)
                    record.executed = True
                    iteration_record["actions"].append({"tool_name": call.func_name, "args": dict(call.args or {})})
                    iteration_record["observations"].append(missing_observation)
                    continue

                execution = await spec.execute(context, dict(call.args or {}), call_id=record.call_id)
                record.executed = True
                record.result = execution
                result.used_tools.append(record.tool_name)
                observation_text = str(execution.content if execution.content is not None else execution.error or "")
                iteration_record["actions"].append({"tool_name": call.func_name, "args": dict(call.args or {})})
                if observation_text.strip():
                    observations.append(observation_text)
                    iteration_record["observations"].append(observation_text)
                tool_message = _build_tool_message(record.call_id, observation_text)
                if tool_message is not None:
                    conversation_messages.append(tool_message)

            loop_state["last_response"] = response or ""
            loop_state["last_reasoning"] = reasoning or ""
            loop_state["pending_observations"] = list(iteration_record["observations"])
            loop_state["last_iteration"] = iteration + 1
            loop_state["last_tool_calls"] = [
                {
                    "name": call.func_name,
                    "args": dict(call.args or {}),
                }
                for call in current_tool_calls
            ]

            if terminal_tool_name is not None:
                break

        result.status = AgentStatus.SUCCESS if terminal_tool_name or observations or result.output_text else AgentStatus.NO_RESULT
        result.observations = observations
        result.terminal_tool_name = terminal_tool_name
        if terminal_tool_args is not None:
            result.source_summary["terminal_tool"] = {
                "name": terminal_tool_name,
                "args": terminal_tool_args,
            }
            if terminal_tool_name == "found_answer":
                result.output_text = str(terminal_tool_args.get("answer", ""))
            elif terminal_tool_name == "not_enough_info":
                result.output_text = str(terminal_tool_args.get("reason", ""))
        return result

    def _build_llm_request(self, profile: AgentProfile):
        from src.config.config import model_config
        from src.llm_models.utils_model import LLMRequest

        return LLMRequest(
            model_set=model_config.model_task_config.get_task(profile.model_task),
            request_type=f"agent.{profile.name}",
        )

    @staticmethod
    def _build_message_list(
        system_prompt: str,
        user_prompt: str,
        conversation_messages: list[Message],
    ) -> list[Message]:
        messages: list[Message] = []
        if system_prompt.strip():
            messages.append(MessageBuilder().set_role(RoleType.System).add_text_content(system_prompt).build())
        if user_prompt.strip():
            messages.append(MessageBuilder().set_role(RoleType.User).add_text_content(user_prompt).build())
        messages.extend(conversation_messages)
        return messages

    @staticmethod
    def _build_single_pass_kwargs(
        llm: Any,
        *,
        prompt: str,
        system_prompt: str | None,
        tools: list[dict[str, Any]] | None,
        reasoning_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "prompt": prompt,
            "tools": tools,
            "system_prompt": system_prompt,
        }
        signature = inspect.signature(llm.generate_response_async)
        if reasoning_config is not None and "reasoning_config" in signature.parameters:
            kwargs["reasoning_config"] = reasoning_config
        return kwargs
