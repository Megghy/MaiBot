from __future__ import annotations

import src.plugin_system.apis as plugin_apis
import src.plugin_system.core as plugin_core


def test_plugin_api_namespace_resolves_modules_lazily():
    assert plugin_apis.send_api.__name__ == "src.plugin_system.apis.send_api"
    assert plugin_apis.frequency_api.__name__ == "src.plugin_system.apis.frequency_api"
    assert callable(plugin_apis.get_logger)


def test_plugin_core_namespace_resolves_compat_exports_lazily():
    assert plugin_core.events_manager is not None
    assert plugin_core.component_registry is not None
