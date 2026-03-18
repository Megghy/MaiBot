"""
记忆检索工具模块
提供统一的工具注册和管理系统
"""

from .tool_registry import (
    MemoryToolSpec,
    clear_registered_memory_tools,
    get_registered_memory_tool,
    get_registered_memory_tools,
    register_memory_retrieval_tool,
)

# 导入所有工具的注册函数
from .query_chat_history import register_tool as register_query_chat_history
from .query_lpmm_knowledge import register_tool as register_lpmm_knowledge
from .query_person_info import register_tool as register_query_person_info
from .query_vector import register_tool as register_query_vector
from src.config.config import global_config
from src.runtime.agent.tool_registry import ToolRegistry


def init_all_tools():
    """初始化并注册所有记忆检索工具"""
    register_query_chat_history()
    register_query_person_info()
    register_query_vector()

    if global_config.lpmm_knowledge.lpmm_mode == "agent":
        register_lpmm_knowledge()


def build_agent_tool_registry() -> ToolRegistry:
    init_all_tools()
    registry = ToolRegistry()
    for tool in get_registered_memory_tools().values():
        registry.register_memory_tool(tool, tags={"memory"})
    return registry


__all__ = [
    "MemoryToolSpec",
    "clear_registered_memory_tools",
    "get_registered_memory_tool",
    "get_registered_memory_tools",
    "register_memory_retrieval_tool",
    "init_all_tools",
    "build_agent_tool_registry",
]
