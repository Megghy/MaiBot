from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Type

from src.common.logger import get_logger
from src.plugin_system.base.plugin_base import PluginBase
from src.plugin_system.core.plugin_manager import plugin_manager
from src.plugins.built_in.emoji_plugin.plugin import CoreActionsPlugin
from src.plugins.built_in.memory_relation_plugin.plugin import MemoryRelationPlugin
from src.plugins.built_in.plugin_management.plugin import PluginManagementPlugin
from src.plugins.built_in.tts_plugin.plugin import TTSPlugin

logger = get_logger("runtime_bootstrap")

BUILT_IN_PLUGIN_CLASSES: tuple[Type[PluginBase], ...] = (
    CoreActionsPlugin,
    MemoryRelationPlugin,
    PluginManagementPlugin,
    TTSPlugin,
)


@dataclass(slots=True)
class RuntimeBootstrapResult:
    builtin_loaded: int
    external_loaded: int
    failed: int

    @property
    def total_loaded(self) -> int:
        return self.builtin_loaded + self.external_loaded


def register_builtin_plugins(plugin_classes: Iterable[Type[PluginBase]] = BUILT_IN_PLUGIN_CLASSES) -> tuple[int, int]:
    return plugin_manager.load_builtin_plugins(plugin_classes)


def bootstrap_runtime() -> RuntimeBootstrapResult:
    builtin_loaded, builtin_failed = register_builtin_plugins()
    external_loaded, external_failed = plugin_manager.load_all_plugins()
    result = RuntimeBootstrapResult(
        builtin_loaded=builtin_loaded,
        external_loaded=external_loaded,
        failed=builtin_failed + external_failed,
    )
    logger.info(
        f"runtime bootstrap 完成: built-in={result.builtin_loaded}, "
        f"external={result.external_loaded}, failed={result.failed}"
    )
    return result
