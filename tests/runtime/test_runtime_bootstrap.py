from __future__ import annotations

from typing import List, Tuple, Type

from src.plugin_system.base.plugin_base import PluginBase
from src.runtime.bootstrap import BUILT_IN_PLUGIN_CLASSES, bootstrap_runtime, register_builtin_plugins


class FakePluginManager:
    def __init__(self) -> None:
        self.builtin_calls: list[Type[PluginBase]] = []
        self.external_called = False

    def load_builtin_plugins(self, plugin_classes: List[Type[PluginBase]]) -> Tuple[int, int]:
        self.builtin_calls = list(plugin_classes)
        return 2, 0

    def load_all_plugins(self) -> Tuple[int, int]:
        self.external_called = True
        return 1, 1


def test_register_builtin_plugins_delegates_to_plugin_manager(monkeypatch):
    fake_manager = FakePluginManager()
    monkeypatch.setattr("src.runtime.bootstrap.plugin_manager", fake_manager)

    loaded, failed = register_builtin_plugins()

    assert loaded == 2
    assert failed == 0
    assert tuple(fake_manager.builtin_calls) == BUILT_IN_PLUGIN_CLASSES


def test_bootstrap_runtime_loads_builtin_then_external(monkeypatch):
    fake_manager = FakePluginManager()
    monkeypatch.setattr("src.runtime.bootstrap.plugin_manager", fake_manager)

    result = bootstrap_runtime()

    assert tuple(fake_manager.builtin_calls) == BUILT_IN_PLUGIN_CLASSES
    assert fake_manager.external_called is True
    assert result.builtin_loaded == 2
    assert result.external_loaded == 1
    assert result.failed == 1
    assert result.total_loaded == 3
