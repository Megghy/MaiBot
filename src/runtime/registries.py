from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional, Pattern, Type, Union

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


ComponentClass = Union[Type[BaseAction], Type[BaseCommand], Type[BaseTool], Type[BaseEventHandler]]


@dataclass(slots=True)
class CommandRegistration:
    info: CommandInfo
    command_class: Type[BaseCommand]
    pattern: Pattern[str]


@dataclass(slots=True)
class EventHandlerRegistration:
    info: EventHandlerInfo
    handler_class: Type[BaseEventHandler]


class RuntimeRegistries:
    def __init__(self) -> None:
        self.plugins: dict[str, PluginInfo] = {}
        self.plugin_configs: dict[str, dict[str, Any]] = {}
        self.components: dict[str, ComponentInfo] = {}
        self.components_by_type: dict[ComponentType, dict[str, ComponentInfo]] = {
            component_type: {} for component_type in ComponentType
        }
        self.component_classes: dict[str, ComponentClass] = {}
        self.actions: dict[str, tuple[ActionInfo, Type[BaseAction]]] = {}
        self.commands: dict[str, CommandRegistration] = {}
        self.command_patterns: dict[Pattern[str], str] = {}
        self.tools: dict[str, tuple[ToolInfo, Type[BaseTool]]] = {}
        self.event_handlers: dict[str, EventHandlerRegistration] = {}

    def _namespaced_name(self, component_type: ComponentType, component_name: str) -> str:
        return f"{component_type.value}.{component_name}"

    def register_plugin(self, plugin_info: PluginInfo, plugin_config: dict[str, Any] | None = None) -> bool:
        if plugin_info.name in self.plugins:
            return False
        self.plugins[plugin_info.name] = plugin_info
        self.plugin_configs[plugin_info.name] = dict(plugin_config or {})
        return True

    def register_component(self, info: ComponentInfo, component_class: ComponentClass) -> bool:
        namespaced_name = self._namespaced_name(info.component_type, info.name)
        if namespaced_name in self.components:
            return False

        self.components[namespaced_name] = info
        self.components_by_type[info.component_type][info.name] = info
        self.component_classes[namespaced_name] = component_class

        match info.component_type:
            case ComponentType.ACTION:
                assert isinstance(info, ActionInfo)
                assert issubclass(component_class, BaseAction)
                self.actions[info.name] = (info, component_class)
            case ComponentType.COMMAND:
                assert isinstance(info, CommandInfo)
                assert issubclass(component_class, BaseCommand)
                pattern = re.compile(info.command_pattern, re.IGNORECASE | re.DOTALL)
                self.commands[info.name] = CommandRegistration(
                    info=info,
                    command_class=component_class,
                    pattern=pattern,
                )
                if info.enabled and pattern not in self.command_patterns:
                    self.command_patterns[pattern] = info.name
            case ComponentType.TOOL:
                assert isinstance(info, ToolInfo)
                assert issubclass(component_class, BaseTool)
                self.tools[info.name] = (info, component_class)
            case ComponentType.EVENT_HANDLER:
                assert isinstance(info, EventHandlerInfo)
                assert issubclass(component_class, BaseEventHandler)
                self.event_handlers[info.name] = EventHandlerRegistration(info=info, handler_class=component_class)
            case _:
                return False
        return True

    def remove_component(self, component_name: str, component_type: ComponentType) -> bool:
        namespaced_name = self._namespaced_name(component_type, component_name)
        info = self.components.pop(namespaced_name, None)
        self.component_classes.pop(namespaced_name, None)
        self.components_by_type[component_type].pop(component_name, None)
        if info is None:
            return False

        match component_type:
            case ComponentType.ACTION:
                self.actions.pop(component_name, None)
            case ComponentType.COMMAND:
                self.commands.pop(component_name, None)
                self.command_patterns = {
                    pattern: name for pattern, name in self.command_patterns.items() if name != component_name
                }
            case ComponentType.TOOL:
                self.tools.pop(component_name, None)
            case ComponentType.EVENT_HANDLER:
                self.event_handlers.pop(component_name, None)
            case _:
                return False
        return True

    def remove_plugin(self, plugin_name: str) -> bool:
        removed = self.plugins.pop(plugin_name, None) is not None
        self.plugin_configs.pop(plugin_name, None)
        return removed

    def enable_component(self, component_name: str, component_type: ComponentType) -> bool:
        info = self.get_component_info(component_name, component_type)
        if info is None:
            return False
        info.enabled = True
        if component_type == ComponentType.COMMAND:
            registration = self.commands.get(component_name)
            if registration:
                self.command_patterns[registration.pattern] = component_name
        return True

    def disable_component(self, component_name: str, component_type: ComponentType) -> bool:
        info = self.get_component_info(component_name, component_type)
        if info is None:
            return False
        info.enabled = False
        if component_type == ComponentType.COMMAND:
            self.command_patterns = {
                pattern: name for pattern, name in self.command_patterns.items() if name != component_name
            }
        return True

    def get_component_info(
        self,
        component_name: str,
        component_type: ComponentType | None = None,
    ) -> Optional[ComponentInfo]:
        if "." in component_name:
            return self.components.get(component_name)
        if component_type is not None:
            return self.components_by_type[component_type].get(component_name)

        matches = [components.get(component_name) for components in self.components_by_type.values()]
        matches = [match for match in matches if match is not None]
        if not matches:
            return None
        return matches[0]

    def get_component_class(
        self,
        component_name: str,
        component_type: ComponentType | None = None,
    ) -> Optional[ComponentClass]:
        if "." in component_name:
            return self.component_classes.get(component_name)
        if component_type is not None:
            return self.component_classes.get(self._namespaced_name(component_type, component_name))

        for current_type in ComponentType:
            namespaced_name = self._namespaced_name(current_type, component_name)
            component_class = self.component_classes.get(namespaced_name)
            if component_class is not None:
                return component_class
        return None

    def get_components_by_type(self, component_type: ComponentType) -> dict[str, ComponentInfo]:
        return self.components_by_type[component_type].copy()

    def get_enabled_components_by_type(self, component_type: ComponentType) -> dict[str, ComponentInfo]:
        return {
            name: info
            for name, info in self.components_by_type[component_type].items()
            if getattr(info, "enabled", False)
        }

    def get_action_registry(self) -> dict[str, Type[BaseAction]]:
        return {name: action_class for name, (_, action_class) in self.actions.items()}

    def get_registered_action_info(self, action_name: str) -> Optional[ActionInfo]:
        item = self.actions.get(action_name)
        return item[0] if item else None

    def get_default_actions(self) -> dict[str, ActionInfo]:
        return {name: info for name, (info, _) in self.actions.items() if getattr(info, "enabled", False)}

    def get_command_registry(self) -> dict[str, Type[BaseCommand]]:
        return {name: registration.command_class for name, registration in self.commands.items()}

    def get_registered_command_info(self, command_name: str) -> Optional[CommandInfo]:
        item = self.commands.get(command_name)
        return item.info if item else None

    def get_command_patterns(self) -> dict[Pattern[str], str]:
        return self.command_patterns.copy()

    def find_command(self, text: str) -> Optional[tuple[Type[BaseCommand], dict[str, str], CommandInfo]]:
        candidates = [pattern for pattern in self.command_patterns if pattern.match(text)]
        if not candidates:
            return None
        command_name = self.command_patterns[candidates[0]]
        registration = self.commands[command_name]
        match = candidates[0].match(text)
        return registration.command_class, match.groupdict() if match else {}, registration.info

    def get_tool_registry(self) -> dict[str, Type[BaseTool]]:
        return {name: tool_class for name, (_, tool_class) in self.tools.items()}

    def get_tool_info(self, tool_name: str) -> Optional[ToolInfo]:
        item = self.tools.get(tool_name)
        return item[0] if item else None

    def get_tool_class(self, tool_name: str) -> Optional[Type[BaseTool]]:
        item = self.tools.get(tool_name)
        return item[1] if item else None

    def get_llm_available_tools(self) -> dict[str, Type[BaseTool]]:
        return {
            name: tool_class
            for name, (tool_info, tool_class) in self.tools.items()
            if getattr(tool_info, "enabled", False)
        }

    def get_registered_tool_info(self, tool_name: str) -> Optional[ToolInfo]:
        return self.get_tool_info(tool_name)

    def get_event_handler_registry(self) -> dict[str, Type[BaseEventHandler]]:
        return {name: registration.handler_class for name, registration in self.event_handlers.items()}

    def get_registered_event_handler_info(self, handler_name: str) -> Optional[EventHandlerInfo]:
        registration = self.event_handlers.get(handler_name)
        return registration.info if registration else None

    def get_enabled_event_handlers(self) -> dict[str, Type[BaseEventHandler]]:
        return {
            name: registration.handler_class
            for name, registration in self.event_handlers.items()
            if getattr(registration.info, "enabled", False)
        }

    def get_enabled_event_handlers_for_event(
        self,
        event_type,
    ) -> list[EventHandlerRegistration]:
        handlers = [
            registration
            for registration in self.event_handlers.values()
            if registration.info.event_type == event_type and getattr(registration.info, "enabled", False)
        ]
        handlers.sort(key=lambda registration: registration.info.weight, reverse=True)
        return handlers

    def get_plugin_info(self, plugin_name: str) -> Optional[PluginInfo]:
        return self.plugins.get(plugin_name)

    def get_all_plugins(self) -> dict[str, PluginInfo]:
        return self.plugins.copy()

    def get_plugin_components(self, plugin_name: str) -> list[ComponentInfo]:
        plugin_info = self.get_plugin_info(plugin_name)
        return plugin_info.components if plugin_info else []

    def get_plugin_config(self, plugin_name: str) -> dict[str, Any] | None:
        plugin_config = self.plugin_configs.get(plugin_name)
        return dict(plugin_config) if plugin_config is not None else None

    def clear(self) -> None:
        self.__init__()

    def get_registry_stats(self) -> dict[str, Any]:
        action_components = len(self.actions)
        command_components = len(self.commands)
        tool_components = len(self.tools)
        event_handlers = len(self.event_handlers)
        total_components = len(self.components)
        enabled_components = sum(1 for component in self.components.values() if getattr(component, "enabled", False))
        enabled_plugins = sum(1 for plugin in self.plugins.values() if getattr(plugin, "enabled", False))

        return {
            "action_components": action_components,
            "command_components": command_components,
            "tool_components": tool_components,
            "event_handlers": event_handlers,
            "total_components": total_components,
            "total_plugins": len(self.plugins),
            "components_by_type": {
                component_type.value: len(components) for component_type, components in self.components_by_type.items()
            },
            "enabled_components": enabled_components,
            "enabled_plugins": enabled_plugins,
            "plugins": len(self.plugins),
            "actions": action_components,
            "commands": command_components,
            "tools": tool_components,
        }


runtime_registries = RuntimeRegistries()
