from __future__ import annotations

import json

from src.plugin_system.base.base_plugin import BasePlugin
from src.plugin_system.base.config_types import ConfigField
from src.plugin_system.core.plugin_manager import PluginManager
from src.runtime.registries import runtime_registries


class DummyPlugin(BasePlugin):
    plugin_name = "dummy_plugin"
    enable_plugin = True
    dependencies = []
    python_dependencies = []
    config_file_name = "config.toml"
    config_schema = {
        "plugin": {
            "enabled": ConfigField(type=bool, default=True, description="启用插件"),
            "config_version": ConfigField(type=str, default="1.0.0", description="配置版本"),
        }
    }

    def get_plugin_components(self):
        return []


def test_plugin_manager_loads_registered_plugin_and_persists_config(tmp_path):
    runtime_registries.clear()

    plugin_dir = tmp_path / "dummy_plugin"
    plugin_dir.mkdir()
    (plugin_dir / "_manifest.json").write_text(
        json.dumps(
            {
                "manifest_version": 1,
                "name": "Dummy Plugin",
                "version": "1.0.0",
                "description": "dummy",
                "author": {"name": "tester"},
            }
        ),
        encoding="utf-8",
    )

    manager = PluginManager(plugin_directories=[])
    manager.built_in_root = tmp_path.resolve()
    manager.register_plugin_class(DummyPlugin, plugin_dir=str(plugin_dir))

    success, count = manager.load_registered_plugin_classes("dummy_plugin")

    assert success is True
    assert count == 1
    assert "dummy_plugin" in manager.loaded_plugins
    assert runtime_registries.get_plugin_info("dummy_plugin") is not None
    assert runtime_registries.get_plugin_info("dummy_plugin").is_built_in is True
    assert runtime_registries.get_plugin_config("dummy_plugin") == {
        "plugin": {"enabled": True, "config_version": "1.0.0"}
    }

    runtime_registries.clear()
