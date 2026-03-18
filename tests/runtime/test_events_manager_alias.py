from src.plugin_system.core import events_manager as package_events_manager
from src.plugin_system.core.events_manager import EventsManager as CompatEventsManager
from src.plugin_system.core.events_manager import events_manager as compat_events_manager
from src.runtime import EventsManager, events_manager
from src.runtime.events import EventsManager as RuntimeEventsManager
from src.runtime.events import events_manager as runtime_events_manager


def test_events_manager_compatibility_exports_share_runtime_singletons():
    assert compat_events_manager is runtime_events_manager
    assert package_events_manager is runtime_events_manager
    assert events_manager is runtime_events_manager
    assert CompatEventsManager is RuntimeEventsManager
    assert EventsManager is RuntimeEventsManager
