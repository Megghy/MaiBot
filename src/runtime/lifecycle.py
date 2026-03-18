import asyncio
from enum import StrEnum

from src.common.logger import get_logger


logger = get_logger("runtime.lifecycle")


class LifecycleAction(StrEnum):
    NONE = "none"
    SHUTDOWN = "shutdown"
    RESTART = "restart"


class LifecycleController:
    def __init__(self) -> None:
        self._requested_action: LifecycleAction = LifecycleAction.NONE
        self._event: asyncio.Event | None = None

    def _ensure_event(self) -> asyncio.Event:
        if self._event is None:
            self._event = asyncio.Event()
        return self._event

    def reset(self) -> None:
        self._requested_action = LifecycleAction.NONE
        self._event = asyncio.Event()

    def request(self, action: LifecycleAction) -> bool:
        if action == LifecycleAction.NONE:
            raise ValueError("不能请求空生命周期动作")

        if self._requested_action != LifecycleAction.NONE:
            logger.warning(f"已有待处理生命周期请求 {self._requested_action}，忽略新的请求 {action}")
            return False

        self._requested_action = action
        self._ensure_event().set()
        logger.info(f"已登记生命周期请求: {action}")
        return True

    async def wait_for_request(self) -> LifecycleAction:
        await self._ensure_event().wait()
        return self._requested_action

    def consume_requested_action(self) -> LifecycleAction:
        action = self._requested_action
        self.reset()
        return action

    @property
    def requested_action(self) -> LifecycleAction:
        return self._requested_action


_controller = LifecycleController()


def reset_lifecycle_state() -> None:
    _controller.reset()


def request_shutdown() -> bool:
    return _controller.request(LifecycleAction.SHUTDOWN)


def request_restart() -> bool:
    return _controller.request(LifecycleAction.RESTART)


async def wait_for_request() -> LifecycleAction:
    return await _controller.wait_for_request()


def consume_requested_action() -> LifecycleAction:
    return _controller.consume_requested_action()


def get_requested_action() -> LifecycleAction:
    return _controller.requested_action
