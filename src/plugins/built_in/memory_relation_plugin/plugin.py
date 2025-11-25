from __future__ import annotations

from typing import List, Tuple, Type

from src.plugin_system import BasePlugin, register_plugin, ComponentInfo
from src.plugin_system.base.config_types import ConfigField
from src.common.logger import get_logger

from .actions import BuildMemoryAction, BuildRelationAction

logger = get_logger("memory_relation_plugin")


@register_plugin
class MemoryRelationPlugin(BasePlugin):
    plugin_name: str = "memory_relation_plugin"
    enable_plugin: bool = True
    config_file_name: str = "config.toml"

    config_section_descriptions = {
        "plugin": "插件启用配置",
        "components": "组件启用配置",
    }

    config_schema = {
        "plugin": {
            "enabled": ConfigField(type=bool, default=True, description="是否启用插件"),
            "config_version": ConfigField(type=str, default="0.1.0", description="配置文件版本"),
        },
        "components": {
            "enable_build_memory": ConfigField(type=bool, default=True, description="启用 build_memory 动作"),
            "enable_build_relation": ConfigField(type=bool, default=True, description="启用 build_relation 动作"),
        },
    }

    def get_plugin_components(self) -> List[Tuple[ComponentInfo, Type]]:
        components: List[Tuple[ComponentInfo, Type]] = []

        if self.get_config("components.enable_build_memory", True):
            components.append((BuildMemoryAction.get_action_info(), BuildMemoryAction))
        if self.get_config("components.enable_build_relation", True):
            components.append((BuildRelationAction.get_action_info(), BuildRelationAction))

        return components
