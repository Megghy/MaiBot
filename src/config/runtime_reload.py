from __future__ import annotations

import copy
import importlib
from dataclasses import dataclass
from typing import Any

from src.common.logger import get_logger
from src.config.config import (
    global_config,
    model_config,
    reload_bot_config_in_place,
    reload_model_config_in_place,
)
from src.llm_models.model_client.base_client import client_registry
from src.runtime.registries import runtime_registries

logger = get_logger("config_reload")


@dataclass(slots=True)
class ConfigReloadResult:
    reloaded: list[str]
    pending_restart: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "reloaded": list(self.reloaded),
            "pending_restart": list(self.pending_restart),
        }


def _ensure_client_types_registered() -> None:
    for provider in model_config.api_providers:
        importlib.import_module(f"src.llm_models.model_client.{provider.client_type}_client")


def _apply_bot_runtime_settings() -> None:
    from src.chat.emoji_system import emoji_manager as emoji_manager_module
    from src.express.expression_learner import expression_learner_manager
    from src.plugins.built_in.emoji_plugin.emoji import EmojiAction
    from src.plugins.built_in.knowledge.lpmm_get_knowledge import SearchKnowledgeFromLPMMTool

    if emoji_manager_module.emoji_manager is not None:
        emoji_manager_module.emoji_manager.refresh_runtime_config()

    expression_learner_manager.refresh_runtime_config()

    EmojiAction.random_activation_probability = global_config.emoji.emoji_chance
    if emoji_action_info := runtime_registries.get_registered_action_info("emoji"):
        emoji_action_info.random_activation_probability = global_config.emoji.emoji_chance

    SearchKnowledgeFromLPMMTool.available_for_llm = global_config.lpmm_knowledge.enable


def _apply_model_runtime_settings() -> None:
    _ensure_client_types_registered()
    client_registry.client_instance_cache.clear()


def _collect_maim_message_pending_restart(previous_config, current_config) -> list[str]:
    pending_restart: list[str] = []

    if previous_config.use_custom != current_config.use_custom:
        pending_restart.append("maim_message.use_custom")
        return pending_restart

    if previous_config.auth_token != current_config.auth_token:
        pending_restart.append("maim_message.auth_token")

    if not current_config.use_custom:
        return pending_restart

    transport_fields = ("host", "port", "mode", "use_wss")
    for field_name in transport_fields:
        if getattr(previous_config, field_name) != getattr(current_config, field_name):
            pending_restart.append(f"maim_message.{field_name}")

    if previous_config.use_wss or current_config.use_wss:
        for field_name in ("cert_file", "key_file"):
            if getattr(previous_config, field_name) != getattr(current_config, field_name):
                pending_restart.append(f"maim_message.{field_name}")

    return pending_restart


def _collect_bot_pending_restart(previous_config, current_config) -> list[str]:
    return _collect_maim_message_pending_restart(previous_config.maim_message, current_config.maim_message)


def hot_reload_bot_config() -> ConfigReloadResult:
    previous_config = copy.deepcopy(global_config)
    reload_bot_config_in_place()
    _apply_bot_runtime_settings()

    pending_restart = _collect_bot_pending_restart(previous_config, global_config)

    logger.info(
        "bot_config 热重载完成" + (f"，仍需重启的配置项: {', '.join(pending_restart)}" if pending_restart else "")
    )
    return ConfigReloadResult(reloaded=["bot_config"], pending_restart=pending_restart)


def hot_reload_model_config() -> ConfigReloadResult:
    reload_model_config_in_place()
    _apply_model_runtime_settings()
    logger.info("model_config 热重载完成")
    return ConfigReloadResult(reloaded=["model_config"], pending_restart=[])


def hot_reload_all_configs() -> ConfigReloadResult:
    bot_result = hot_reload_bot_config()
    model_result = hot_reload_model_config()
    return ConfigReloadResult(
        reloaded=bot_result.reloaded + model_result.reloaded,
        pending_restart=list(dict.fromkeys(bot_result.pending_restart + model_result.pending_restart)),
    )
