from __future__ import annotations

import importlib

from src.plugin_system.base.component_types import ComponentType
from src.plugin_system.core.global_announcement_manager import GlobalAnnouncementManager
from src.runtime.local_component_state import LocalComponentState

announcement_module = importlib.import_module("src.plugin_system.core.global_announcement_manager")


def test_local_component_state_tracks_disabled_components():
    state = LocalComponentState()

    assert state.disable_component("stream-1", "reply", ComponentType.ACTION) is True
    assert state.disable_component("stream-1", "reply", ComponentType.ACTION) is False
    assert state.get_disabled_components("stream-1", ComponentType.ACTION) == ["reply"]
    assert state.enable_component("stream-1", "reply", ComponentType.ACTION) is True
    assert state.get_disabled_components("stream-1", ComponentType.ACTION) == []


def test_global_announcement_manager_delegates_to_runtime_state(monkeypatch):
    state = LocalComponentState()
    monkeypatch.setattr(announcement_module, "local_component_state", state)

    manager = GlobalAnnouncementManager()

    assert manager.disable_specific_chat_command("stream-1", "management") is True
    assert manager.get_disabled_chat_commands("stream-1") == ["management"]
    assert manager.enable_specific_chat_command("stream-1", "management") is True
    assert manager.get_disabled_chat_commands("stream-1") == []
