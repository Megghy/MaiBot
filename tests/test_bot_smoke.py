from __future__ import annotations

import asyncio
import importlib

import pytest

from src.plugin_system.base.component_types import EventType


@pytest.fixture(scope="module")
def bot_module():
    return importlib.import_module("bot")


def test_raw_main_runs_startup_boundary(monkeypatch, bot_module):
    calls: list[str] = []
    fake_system = object()

    monkeypatch.setattr(bot_module.platform, "system", lambda: "Windows")
    monkeypatch.setattr(bot_module, "check_eula", lambda: calls.append("check_eula"))
    monkeypatch.setattr(bot_module, "easter_egg", lambda: calls.append("easter_egg"))
    monkeypatch.setattr(bot_module, "MainSystem", lambda: fake_system)

    result = bot_module.raw_main()

    assert result is fake_system
    assert calls == ["check_eula", "easter_egg"]


@pytest.mark.asyncio
async def test_graceful_shutdown_runs_runtime_and_server_cleanup(monkeypatch, bot_module):
    calls: list[object] = []

    class FakeRuntimeManager:
        async def shutdown(self):
            calls.append("runtime_shutdown")

    class FakeServer:
        async def shutdown(self):
            calls.append("server_shutdown")

    class FakeWebUIServer:
        _server = object()

        async def shutdown(self):
            calls.append("webui_shutdown")

    class FakeEventsManager:
        async def handle_mai_events(self, *, event_type):
            calls.append(("event", event_type))
            return True, None

    async def fake_stop_tasks():
        calls.append("stop_tasks")

    import src.runtime as runtime_module
    import src.common.server as server_module
    import src.webui.webui_server as webui_module

    monkeypatch.setattr(runtime_module, "events_manager", FakeEventsManager(), raising=False)
    monkeypatch.setattr(runtime_module, "get_runtime_manager", lambda: FakeRuntimeManager(), raising=False)
    monkeypatch.setattr(server_module, "get_global_server", lambda: FakeServer())
    monkeypatch.setattr(webui_module, "get_webui_server", lambda: FakeWebUIServer())
    monkeypatch.setattr(bot_module.async_task_manager, "stop_and_wait_all_tasks", fake_stop_tasks)
    monkeypatch.setattr(bot_module.asyncio, "all_tasks", lambda: {asyncio.current_task()})

    await bot_module.graceful_shutdown()

    assert ("event", EventType.ON_STOP) in calls
    assert "runtime_shutdown" in calls
    assert "stop_tasks" in calls
    assert "server_shutdown" in calls
    assert "webui_shutdown" in calls
