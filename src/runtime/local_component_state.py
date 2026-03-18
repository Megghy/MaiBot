from __future__ import annotations

from src.common.logger import get_logger
from src.plugin_system.base.component_types import ComponentType

logger = get_logger("runtime.local_component_state")


class LocalComponentState:
    def __init__(self) -> None:
        self._disabled_components: dict[ComponentType, dict[str, list[str]]] = {
            ComponentType.ACTION: {},
            ComponentType.COMMAND: {},
            ComponentType.EVENT_HANDLER: {},
            ComponentType.TOOL: {},
        }

    def disable_component(self, stream_id: str, component_name: str, component_type: ComponentType) -> bool:
        disabled_components = self._disabled_components[component_type].setdefault(stream_id, [])
        if component_name in disabled_components:
            logger.warning(f"{component_type.value} {component_name} 已经被禁用")
            return False
        disabled_components.append(component_name)
        return True

    def enable_component(self, stream_id: str, component_name: str, component_type: ComponentType) -> bool:
        disabled_components = self._disabled_components[component_type].get(stream_id)
        if not disabled_components:
            return False
        try:
            disabled_components.remove(component_name)
            return True
        except ValueError:
            logger.warning(f"{component_type.value} {component_name} 不在禁用列表中")
            return False

    def get_disabled_components(self, stream_id: str, component_type: ComponentType) -> list[str]:
        return self._disabled_components[component_type].get(stream_id, []).copy()


local_component_state = LocalComponentState()
