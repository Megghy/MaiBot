from __future__ import annotations

from typing import List

from src.common.logger import get_logger
from src.plugin_system.base.component_types import ComponentType
from src.runtime.local_component_state import local_component_state

logger = get_logger("global_announcement_manager")


class GlobalAnnouncementManager:
    """本地启停状态的兼容壳。"""

    def __init__(self) -> None:
        logger.debug("global_announcement_manager 处于兼容模式，状态已委托给 runtime.local_component_state")

    def disable_specific_chat_action(self, chat_id: str, action_name: str) -> bool:
        return local_component_state.disable_component(chat_id, action_name, ComponentType.ACTION)

    def enable_specific_chat_action(self, chat_id: str, action_name: str) -> bool:
        return local_component_state.enable_component(chat_id, action_name, ComponentType.ACTION)

    def disable_specific_chat_command(self, chat_id: str, command_name: str) -> bool:
        return local_component_state.disable_component(chat_id, command_name, ComponentType.COMMAND)

    def enable_specific_chat_command(self, chat_id: str, command_name: str) -> bool:
        return local_component_state.enable_component(chat_id, command_name, ComponentType.COMMAND)

    def disable_specific_chat_event_handler(self, chat_id: str, handler_name: str) -> bool:
        return local_component_state.disable_component(chat_id, handler_name, ComponentType.EVENT_HANDLER)

    def enable_specific_chat_event_handler(self, chat_id: str, handler_name: str) -> bool:
        return local_component_state.enable_component(chat_id, handler_name, ComponentType.EVENT_HANDLER)

    def disable_specific_chat_tool(self, chat_id: str, tool_name: str) -> bool:
        return local_component_state.disable_component(chat_id, tool_name, ComponentType.TOOL)

    def enable_specific_chat_tool(self, chat_id: str, tool_name: str) -> bool:
        return local_component_state.enable_component(chat_id, tool_name, ComponentType.TOOL)

    def get_disabled_chat_actions(self, chat_id: str) -> List[str]:
        return local_component_state.get_disabled_components(chat_id, ComponentType.ACTION)

    def get_disabled_chat_commands(self, chat_id: str) -> List[str]:
        return local_component_state.get_disabled_components(chat_id, ComponentType.COMMAND)

    def get_disabled_chat_event_handlers(self, chat_id: str) -> List[str]:
        return local_component_state.get_disabled_components(chat_id, ComponentType.EVENT_HANDLER)

    def get_disabled_chat_tools(self, chat_id: str) -> List[str]:
        return local_component_state.get_disabled_components(chat_id, ComponentType.TOOL)


global_announcement_manager = GlobalAnnouncementManager()
