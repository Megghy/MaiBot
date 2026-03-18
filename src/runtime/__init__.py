from src.runtime.command_registry import CommandRegistry
from src.runtime.event_bus import EventBus
from src.runtime.models import CommandExecutionResult

__all__ = [
    "bootstrap_runtime",
    "CommandExecutionResult",
    "CommandRegistry",
    "EventsManager",
    "EventBus",
    "events_manager",
    "local_component_state",
    "LocalComponentState",
    "MessagePipeline",
    "RuntimeManager",
    "RuntimeRegistries",
    "SessionRuntime",
    "get_runtime_manager",
    "runtime_manager",
    "runtime_registries",
    "set_runtime_manager",
]


def __getattr__(name: str):
    if name == "bootstrap_runtime":
        from src.runtime.bootstrap import bootstrap_runtime

        return bootstrap_runtime

    if name in {"RuntimeManager", "get_runtime_manager", "runtime_manager", "set_runtime_manager"}:
        from src.runtime.manager import RuntimeManager, get_runtime_manager, runtime_manager, set_runtime_manager

        mapping = {
            "RuntimeManager": RuntimeManager,
            "get_runtime_manager": get_runtime_manager,
            "runtime_manager": runtime_manager,
            "set_runtime_manager": set_runtime_manager,
        }
        return mapping[name]

    if name in {"RuntimeRegistries", "runtime_registries"}:
        from src.runtime.registries import RuntimeRegistries, runtime_registries

        mapping = {
            "RuntimeRegistries": RuntimeRegistries,
            "runtime_registries": runtime_registries,
        }
        return mapping[name]

    if name == "SessionRuntime":
        from src.runtime.session import SessionRuntime

        return SessionRuntime

    if name == "MessagePipeline":
        from src.runtime.message_pipeline import MessagePipeline

        return MessagePipeline

    if name in {"EventsManager", "events_manager"}:
        from src.runtime.events import EventsManager, events_manager

        mapping = {
            "EventsManager": EventsManager,
            "events_manager": events_manager,
        }
        return mapping[name]

    if name in {"LocalComponentState", "local_component_state"}:
        from src.runtime.local_component_state import LocalComponentState, local_component_state

        mapping = {
            "LocalComponentState": LocalComponentState,
            "local_component_state": local_component_state,
        }
        return mapping[name]

    raise AttributeError(f"module 'src.runtime' has no attribute {name!r}")
