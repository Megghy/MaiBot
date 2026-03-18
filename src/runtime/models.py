from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from src.chat.message_receive.chat_stream import ChatStream
from src.common.data_models.database_data_model import DatabaseMessages


@dataclass(slots=True)
class SessionState:
    chat_stream: ChatStream
    controller: Any
    pending_messages: list[DatabaseMessages] = field(default_factory=list)
    last_message: DatabaseMessages | None = None
    flush_task: asyncio.Task | None = None
    initialized: bool = False
    runtime_objects: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CommandExecutionResult:
    name: str
    handled: bool
    intercepted: bool
    response: str | None = None
