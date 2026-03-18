from __future__ import annotations

from dataclasses import dataclass, field

from src.config.config import _sync_config_object_in_place
from src.config.config_base import ConfigBase
from src.config.api_ada_configs import TaskConfig
from src.config.official_configs import MaimMessageConfig
from src.config.runtime_reload import _collect_maim_message_pending_restart
from src.llm_models.utils_model import LLMRequest


@dataclass
class ChildConfig(ConfigBase):
    value: int


@dataclass
class RootConfig(ConfigBase):
    child: ChildConfig
    numbers: list[int] = field(default_factory=list)
    flags: set[str] = field(default_factory=set)
    mapping: dict[str, int] = field(default_factory=dict)


def test_sync_config_object_in_place_preserves_nested_identity():
    current = RootConfig(
        child=ChildConfig(value=1),
        numbers=[1],
        flags={"a"},
        mapping={"x": 1},
    )
    original_child = current.child

    updated = RootConfig(
        child=ChildConfig(value=2),
        numbers=[2, 3],
        flags={"b"},
        mapping={"y": 2},
    )

    synced = _sync_config_object_in_place(current, updated)

    assert synced is current
    assert current.child is original_child
    assert current.child.value == 2
    assert current.numbers == [2, 3]
    assert current.flags == {"b"}
    assert current.mapping == {"y": 2}


def test_llm_request_sync_model_usage_state_tracks_new_model_list():
    task_config = TaskConfig(model_list=["model-a"])
    request = LLMRequest(model_set=task_config, request_type="test")
    request.model_usage["model-a"] = (10, 1, 0)

    task_config.model_list[:] = ["model-b", "model-c"]
    request.sync_model_usage_state()

    assert request.model_usage == {
        "model-b": (0, 0, 0),
        "model-c": (0, 0, 0),
    }


def test_collect_maim_message_pending_restart_ignores_inactive_custom_transport_changes():
    previous = MaimMessageConfig(use_custom=False, host="127.0.0.1", port=8090)
    current = MaimMessageConfig(use_custom=False, host="0.0.0.0", port=9000)

    assert _collect_maim_message_pending_restart(previous, current) == []


def test_collect_maim_message_pending_restart_tracks_active_transport_changes():
    previous = MaimMessageConfig(use_custom=True, host="127.0.0.1", port=8090, mode="ws", use_wss=False)
    current = MaimMessageConfig(use_custom=True, host="0.0.0.0", port=9000, mode="tcp", use_wss=True)

    assert _collect_maim_message_pending_restart(previous, current) == [
        "maim_message.host",
        "maim_message.port",
        "maim_message.mode",
        "maim_message.use_wss",
    ]


def test_collect_maim_message_pending_restart_tracks_use_custom_and_auth_token():
    previous = MaimMessageConfig(use_custom=False, auth_token=["a"])
    current = MaimMessageConfig(use_custom=True, auth_token=["b"])

    assert _collect_maim_message_pending_restart(previous, current) == ["maim_message.use_custom"]

    token_only_current = MaimMessageConfig(use_custom=False, auth_token=["b"])
    assert _collect_maim_message_pending_restart(previous, token_only_current) == ["maim_message.auth_token"]
