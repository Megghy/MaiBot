"""Legacy plugin-system namespace.

Prefer importing concrete symbols from:
- `src.plugin_system.base`
- `src.plugin_system.apis`
- `src.plugin_system.core`
"""

__version__ = "2.0.0"

__all__ = [
    "component_registry",
    "events_manager",
    "global_announcement_manager",
    "plugin_manager",
    "runtime_registries",
]


def __getattr__(name: str):
    if name in {"component_registry", "events_manager", "global_announcement_manager", "plugin_manager"}:
        from .core import component_registry, events_manager, global_announcement_manager, plugin_manager

        mapping = {
            "component_registry": component_registry,
            "events_manager": events_manager,
            "global_announcement_manager": global_announcement_manager,
            "plugin_manager": plugin_manager,
        }
        return mapping[name]

    if name == "runtime_registries":
        from src.runtime import runtime_registries

        return runtime_registries

    raise AttributeError(f"module 'src.plugin_system' has no attribute {name!r}")
