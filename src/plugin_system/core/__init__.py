"""Legacy plugin-system core namespace."""

from src.plugin_system.core.component_registry import component_registry
from src.plugin_system.core.events_manager import events_manager
from src.plugin_system.core.global_announcement_manager import global_announcement_manager
from src.plugin_system.core.plugin_manager import plugin_manager

__all__ = [
    "plugin_manager",
    "component_registry",
    "events_manager",
    "global_announcement_manager",
]
