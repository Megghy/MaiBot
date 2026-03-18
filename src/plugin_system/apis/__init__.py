"""Plugin API namespace.

Keep imports narrow where possible, for example:
- `from src.plugin_system.apis import send_api`
- `from src.plugin_system.apis.logging_api import get_logger`
"""

__all__ = [
    "component_manage_api",
    "emoji_api",
    "frequency_api",
    "generator_api",
    "get_logger",
    "plugin_manage_api",
    "send_api",
]


def __getattr__(name: str):
    if name == "get_logger":
        from .logging_api import get_logger

        return get_logger

    module_names = {
        "auto_talk_api",
        "chat_api",
        "component_manage_api",
        "config_api",
        "database_api",
        "emoji_api",
        "frequency_api",
        "generator_api",
        "llm_api",
        "message_api",
        "mood_api",
        "person_api",
        "plugin_manage_api",
        "send_api",
        "tool_api",
    }
    if name in module_names:
        module = __import__(f"src.plugin_system.apis.{name}", fromlist=[name])
        return module

    raise AttributeError(f"module 'src.plugin_system.apis' has no attribute {name!r}")
