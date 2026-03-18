from __future__ import annotations

import pytest

from src.plugin_system.base.base_events_handler import BaseEventHandler
from src.plugin_system.base.component_types import EventType, PluginInfo
from src.plugin_system.core.component_registry import component_registry
from src.runtime.events import EventsManager
from src.runtime.registries import runtime_registries


class RecordingStopHandler(BaseEventHandler):
    event_type = EventType.ON_STOP
    handler_name = "record_stop"
    intercept_message = True
    seen_plugin_name = None
    seen_plugin_config = None

    async def execute(self, message):
        del message
        type(self).seen_plugin_name = self.plugin_name
        type(self).seen_plugin_config = self.plugin_config
        return True, True, "ok", None, None


@pytest.mark.asyncio
async def test_events_manager_dispatches_registered_handler_with_plugin_config():
    runtime_registries.clear()
    RecordingStopHandler.seen_plugin_name = None
    RecordingStopHandler.seen_plugin_config = None

    handler_info = RecordingStopHandler.get_handler_info()
    handler_info.plugin_name = "test_plugin"
    plugin_info = PluginInfo(
        display_name="Test Plugin",
        name="test_plugin",
        description="runtime events test",
        components=[handler_info],
    )

    try:
        assert component_registry.register_plugin(plugin_info, {"feature": {"enabled": True}}) is True
        assert component_registry.register_component(handler_info, RecordingStopHandler) is True

        manager = EventsManager()
        continue_flag, modified_message = await manager.handle_mai_events(EventType.ON_STOP)

        assert continue_flag is True
        assert modified_message is None
        assert RecordingStopHandler.seen_plugin_name == "test_plugin"
        assert RecordingStopHandler.seen_plugin_config == {"feature": {"enabled": True}}
    finally:
        runtime_registries.clear()
