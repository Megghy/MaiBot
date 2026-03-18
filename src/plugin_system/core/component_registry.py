from __future__ import annotations

from typing import Any, Dict, Optional, Pattern, Type, Union

from src.common.logger import get_logger
from src.plugin_system.base.base_action import BaseAction
from src.plugin_system.base.base_command import BaseCommand
from src.plugin_system.base.base_events_handler import BaseEventHandler
from src.plugin_system.base.base_tool import BaseTool
from src.plugin_system.base.component_types import (
    ActionInfo,
    CommandInfo,
    ComponentInfo,
    ComponentType,
    EventHandlerInfo,
    PluginInfo,
    ToolInfo,
)
from src.runtime.registries import runtime_registries

logger = get_logger("component_registry")


class ComponentRegistry:
    """兼容门面：活动注册数据全部委托给 runtime_registries。"""

    def __init__(self) -> None:
        logger.info("组件注册中心初始化完成")

    def register_plugin(self, plugin_info: PluginInfo, plugin_config: dict[str, Any] | None = None) -> bool:
        if not runtime_registries.register_plugin(plugin_info, plugin_config):
            logger.warning(f"插件 {plugin_info.name} 已存在，跳过注册")
            return False
        logger.debug(f"已注册插件: {plugin_info.name} (组件数量: {len(plugin_info.components)})")
        return True

    def register_component(
        self,
        component_info: ComponentInfo,
        component_class: Type[Union[BaseCommand, BaseAction, BaseEventHandler, BaseTool]],
    ) -> bool:
        component_name = component_info.name
        plugin_name = getattr(component_info, "plugin_name", "unknown")
        if "." in component_name:
            logger.error(f"组件名称 '{component_name}' 包含非法字符 '.'，请使用下划线替代")
            return False
        if "." in plugin_name:
            logger.error(f"插件名称 '{plugin_name}' 包含非法字符 '.'，请使用下划线替代")
            return False

        if not runtime_registries.register_component(component_info, component_class):
            logger.warning(f"组件名冲突: '{plugin_name}' 的组件 '{component_name}' 已注册，跳过")
            return False

        logger.debug(
            f"已注册{component_info.component_type}组件: '{component_name}' ({component_class.__name__}) [插件: {plugin_name}]"
        )
        return True

    async def remove_component(self, component_name: str, component_type: ComponentType, plugin_name: str) -> bool:
        removed = runtime_registries.remove_component(component_name, component_type)
        if removed:
            logger.info(f"组件 {plugin_name}.{component_name} 已移除")
        else:
            logger.warning(f"组件 {plugin_name}.{component_name} 未注册，无法移除")
        return removed

    def remove_plugin_registry(self, plugin_name: str) -> bool:
        removed = runtime_registries.remove_plugin(plugin_name)
        if removed:
            logger.info(f"插件 {plugin_name} 已移除")
        else:
            logger.warning(f"插件 {plugin_name} 未注册，无法移除")
        return removed

    def enable_component(self, component_name: str, component_type: ComponentType) -> bool:
        enabled = runtime_registries.enable_component(component_name, component_type)
        if enabled:
            logger.info(f"组件 {component_name} 已启用")
        else:
            logger.warning(f"组件 {component_name} 未注册，无法启用")
        return enabled

    async def disable_component(self, component_name: str, component_type: ComponentType) -> bool:
        disabled = runtime_registries.disable_component(component_name, component_type)
        if disabled:
            logger.info(f"组件 {component_name} 已禁用")
        else:
            logger.warning(f"组件 {component_name} 未注册，无法禁用")
        return disabled

    def get_component_info(
        self,
        component_name: str,
        component_type: Optional[ComponentType] = None,
    ) -> Optional[ComponentInfo]:
        return runtime_registries.get_component_info(component_name, component_type)

    def get_component_class(
        self,
        component_name: str,
        component_type: Optional[ComponentType] = None,
    ) -> Optional[Union[Type[BaseCommand], Type[BaseAction], Type[BaseEventHandler], Type[BaseTool]]]:
        return runtime_registries.get_component_class(component_name, component_type)

    def get_components_by_type(self, component_type: ComponentType) -> Dict[str, ComponentInfo]:
        return runtime_registries.get_components_by_type(component_type)

    def get_enabled_components_by_type(self, component_type: ComponentType) -> Dict[str, ComponentInfo]:
        return runtime_registries.get_enabled_components_by_type(component_type)

    def get_action_registry(self) -> Dict[str, Type[BaseAction]]:
        return runtime_registries.get_action_registry()

    def get_registered_action_info(self, action_name: str) -> Optional[ActionInfo]:
        return runtime_registries.get_registered_action_info(action_name)

    def get_default_actions(self) -> Dict[str, ActionInfo]:
        return runtime_registries.get_default_actions()

    def get_command_registry(self) -> Dict[str, Type[BaseCommand]]:
        return runtime_registries.get_command_registry()

    def get_registered_command_info(self, command_name: str) -> Optional[CommandInfo]:
        return runtime_registries.get_registered_command_info(command_name)

    def get_command_patterns(self) -> Dict[Pattern, str]:
        return runtime_registries.get_command_patterns()

    def find_command_by_text(self, text: str) -> Optional[tuple[Type[BaseCommand], dict, CommandInfo]]:
        return runtime_registries.find_command(text)

    def get_tool_registry(self) -> Dict[str, Type[BaseTool]]:
        return runtime_registries.get_tool_registry()

    def get_llm_available_tools(self) -> Dict[str, Type[BaseTool]]:
        return runtime_registries.get_llm_available_tools()

    def get_registered_tool_info(self, tool_name: str) -> Optional[ToolInfo]:
        return runtime_registries.get_registered_tool_info(tool_name)

    def get_event_handler_registry(self) -> Dict[str, Type[BaseEventHandler]]:
        return runtime_registries.get_event_handler_registry()

    def get_registered_event_handler_info(self, handler_name: str) -> Optional[EventHandlerInfo]:
        return runtime_registries.get_registered_event_handler_info(handler_name)

    def get_enabled_event_handlers(self) -> Dict[str, Type[BaseEventHandler]]:
        return runtime_registries.get_enabled_event_handlers()

    def get_plugin_info(self, plugin_name: str) -> Optional[PluginInfo]:
        return runtime_registries.get_plugin_info(plugin_name)

    def get_all_plugins(self) -> Dict[str, PluginInfo]:
        return runtime_registries.get_all_plugins()

    def get_plugin_components(self, plugin_name: str) -> list[ComponentInfo]:
        return runtime_registries.get_plugin_components(plugin_name)

    def get_plugin_config(self, plugin_name: str) -> Optional[dict]:
        return runtime_registries.get_plugin_config(plugin_name)

    def get_registry_stats(self) -> Dict[str, Any]:
        return runtime_registries.get_registry_stats()


component_registry = ComponentRegistry()
