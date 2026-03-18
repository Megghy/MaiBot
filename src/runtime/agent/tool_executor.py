from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from src.chat.message_receive.chat_stream import get_chat_manager
from src.chat.utils.prompt_builder import Prompt, global_prompt_manager
from src.common.logger import get_logger
from src.config.config import global_config, model_config
from src.llm_models.payload_content import ToolCall
from src.llm_models.utils_model import LLMRequest
from src.plugin_system.apis.tool_api import get_llm_available_tool_definitions, get_tool_instance
from src.plugin_system.base.base_tool import BaseTool
from src.plugin_system.base.component_types import ComponentType
from src.runtime.agent import AgentMode, AgentProfile, AgentRequest, AgentRunner, PromptBlock, PromptBlockRole
from src.runtime.local_component_state import local_component_state

logger = get_logger("agent_tool_executor")


def init_tool_executor_prompt() -> None:
    Prompt(
        """
你是一个专门执行工具的助手。你的名字是{bot_name}。现在是{time_now}。
群里正在进行的聊天内容：
{chat_history}

现在，{sender}发送了内容:{target_message},你想要回复ta。
请仔细分析聊天内容，考虑以下几点：
1. 内容中是否包含需要查询信息的问题
2. 是否有明确的工具使用指令
你可以选择多个动作

If you need to use tools, please directly call the corresponding tool function. If you do not need to use any tool, simply output "No tool needed".
""",
        "tool_executor_prompt",
    )


init_tool_executor_prompt()


class AgentToolExecutor:
    def __init__(self, chat_id: str, enable_cache: bool = True, cache_ttl: int = 3):
        self.chat_id = chat_id
        self.chat_stream = get_chat_manager().get_stream(self.chat_id)
        self.log_prefix = f"[{get_chat_manager().get_stream_name(self.chat_id) or self.chat_id}]"
        self.llm_model = LLMRequest(model_set=model_config.model_task_config.tool_use, request_type="tool_executor")
        self.enable_cache = enable_cache
        self.cache_ttl = cache_ttl
        self.tool_cache: dict[str, dict[str, Any]] = {}

    async def execute_from_chat_message(
        self, target_message: str, chat_history: str, sender: str, return_details: bool = False
    ) -> Tuple[List[Dict[str, Any]], List[str], str]:
        cache_key = self._generate_cache_key(target_message, chat_history, sender)
        if cached_result := self._get_from_cache(cache_key):
            if not return_details:
                return cached_result, [], ""
            used_tools = [result.get("tool_name", "unknown") for result in cached_result]
            return cached_result, used_tools, ""

        tools = self._get_tool_definitions()
        if not tools:
            return [], [], ""

        prompt = await global_prompt_manager.format_prompt(
            "tool_executor_prompt",
            target_message=target_message,
            chat_history=chat_history,
            sender=sender,
            bot_name=global_config.bot.nickname,
            time_now=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        )
        agent_result = await AgentRunner().run(
            AgentRequest(
                request_id=f"tool-executor:{self.chat_id}:{int(time.time() * 1000)}",
                chat_id=self.chat_id,
                user_id=None,
                message=target_message,
                skill_name="tool_only",
                prompt_blocks=[
                    PromptBlock(name="task", content=prompt, role=PromptBlockRole.USER),
                ],
                metadata={"tool_definitions": tools},
            ),
            AgentProfile(
                name="tool_only",
                skill_name="tool_only",
                agent_mode=AgentMode.SINGLE_PASS,
                model_task="tool_use",
            ),
            llm_request=self.llm_model,
        )
        tool_calls = [ToolCall(record.call_id, record.tool_name, record.args) for record in agent_result.tool_calls]
        tool_results, used_tools = await self.execute_tool_calls(tool_calls)
        if tool_results:
            self._set_cache(cache_key, tool_results)
        if return_details:
            return tool_results, used_tools, prompt
        return tool_results, [], ""

    def _get_tool_definitions(self) -> List[Dict[str, Any]]:
        all_tools = get_llm_available_tool_definitions()
        user_disabled_tools = local_component_state.get_disabled_components(self.chat_id, ComponentType.TOOL)
        return [definition for name, definition in all_tools if name not in user_disabled_tools]

    async def execute_tool_calls(self, tool_calls: Optional[List[ToolCall]]) -> Tuple[List[Dict[str, Any]], List[str]]:
        tool_results: List[Dict[str, Any]] = []
        used_tools: List[str] = []
        if not tool_calls:
            return [], []

        for tool_call in tool_calls:
            tool_name = tool_call.func_name
            try:
                result = await self.execute_tool_call(tool_call)
                if not result:
                    continue
                tool_info = {
                    "type": result.get("type", "unknown_type"),
                    "id": result.get("id", f"tool_exec_{time.time()}"),
                    "content": result.get("content", ""),
                    "tool_name": tool_name,
                    "timestamp": time.time(),
                }
                content = tool_info["content"]
                if not isinstance(content, (str, list, tuple)):
                    tool_info["content"] = str(content)
                normalized_content = tool_info["content"]
                if (isinstance(normalized_content, str) and not normalized_content.strip()) or (
                    isinstance(normalized_content, (list, tuple)) and len(normalized_content) == 0
                ):
                    continue
                tool_results.append(tool_info)
                used_tools.append(tool_name)
            except Exception as exc:
                logger.error(f"{self.log_prefix}工具{tool_name}执行失败: {exc}")
                tool_results.append(
                    {
                        "type": "tool_error",
                        "id": f"tool_error_{time.time()}",
                        "content": f"工具{tool_name}执行失败: {exc}",
                        "tool_name": tool_name,
                        "timestamp": time.time(),
                    }
                )

        return tool_results, used_tools

    async def execute_tool_call(
        self, tool_call: ToolCall, tool_instance: Optional[BaseTool] = None
    ) -> Optional[Dict[str, Any]]:
        function_name = tool_call.func_name
        function_args = dict(tool_call.args or {})
        function_args["llm_called"] = True
        tool_instance = tool_instance or get_tool_instance(function_name, self.chat_stream)
        if not tool_instance:
            logger.warning(f"未知工具名称: {function_name}")
            return None
        result = await tool_instance.execute(function_args)
        if not result:
            return None
        return {
            "tool_call_id": tool_call.call_id,
            "role": "tool",
            "name": function_name,
            "type": "function",
            "content": result["content"],
        }

    async def execute_specific_tool_simple(self, tool_name: str, tool_args: Dict) -> Optional[Dict]:
        tool_call = ToolCall(
            call_id=f"direct_tool_{time.time()}",
            func_name=tool_name,
            args=tool_args,
        )
        result = await self.execute_tool_call(tool_call)
        if not result:
            return None
        return {
            "type": result.get("type", "unknown_type"),
            "id": result.get("id", f"direct_tool_{time.time()}"),
            "content": result.get("content", ""),
            "tool_name": tool_name,
            "timestamp": time.time(),
        }

    def _generate_cache_key(self, target_message: str, chat_history: str, sender: str) -> str:
        import hashlib

        return hashlib.md5(f"{target_message}_{chat_history}_{sender}".encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[List[Dict]]:
        if not self.enable_cache or cache_key not in self.tool_cache:
            return None
        cache_item = self.tool_cache[cache_key]
        if cache_item["ttl"] <= 0:
            del self.tool_cache[cache_key]
            return None
        cache_item["ttl"] -= 1
        return cache_item["result"]

    def _set_cache(self, cache_key: str, result: List[Dict]) -> None:
        if not self.enable_cache:
            return
        self.tool_cache[cache_key] = {"result": result, "ttl": self.cache_ttl, "timestamp": time.time()}

    def clear_cache(self) -> None:
        self.tool_cache.clear()

    def get_cache_status(self) -> Dict:
        if not self.enable_cache:
            return {"enabled": False, "cache_count": 0}
        return {
            "enabled": True,
            "cache_count": len(self.tool_cache),
            "cache_ttl": self.cache_ttl,
        }

    def set_cache_config(self, enable_cache: Optional[bool] = None, cache_ttl: int = -1) -> None:
        if enable_cache is not None:
            self.enable_cache = enable_cache
        if cache_ttl > 0:
            self.cache_ttl = cache_ttl


ToolExecutor = AgentToolExecutor

__all__ = ["AgentToolExecutor", "ToolExecutor", "init_tool_executor_prompt"]
