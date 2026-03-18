from __future__ import annotations

from textwrap import dedent

from src.plugin_system.core.plugin_manager import PluginManager


def test_plugin_manager_discovers_external_plugin_via_explicit_plugin_class(tmp_path):
    plugins_root = tmp_path / "plugins"
    plugin_dir = plugins_root / "sample_plugin"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.py").write_text(
        dedent(
            """
            from src.plugin_system.base.base_plugin import BasePlugin


            class SamplePlugin(BasePlugin):
                plugin_name = "sample_plugin"


            PLUGIN_CLASS = SamplePlugin
            """
        ),
        encoding="utf-8",
    )

    manager = PluginManager(plugin_directories=[str(plugins_root)])

    discovered, failed = manager.rescan_plugin_directory()

    assert discovered == 1
    assert failed == 0
    assert manager.list_registered_plugins() == ["sample_plugin"]
    assert manager.get_plugin_path("sample_plugin") == str(plugin_dir.resolve())


def test_plugin_manager_rejects_external_plugin_without_explicit_plugin_class(tmp_path):
    plugins_root = tmp_path / "plugins"
    plugin_dir = plugins_root / "broken_plugin"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.py").write_text(
        dedent(
            """
            from src.plugin_system.base.base_plugin import BasePlugin


            class BrokenPlugin(BasePlugin):
                plugin_name = "broken_plugin"
            """
        ),
        encoding="utf-8",
    )

    manager = PluginManager(plugin_directories=[str(plugins_root)])

    discovered, failed = manager.rescan_plugin_directory()

    assert discovered == 0
    assert failed == 1
    assert manager.list_registered_plugins() == []
    assert "broken_plugin" in manager.failed_plugins
