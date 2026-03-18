from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.runtime.agent.models import PromptBlock, PromptBlockRole


@dataclass(slots=True)
class AssembledContext:
    blocks: list[PromptBlock] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class ContextAssembler:
    """Small structured context helper for the first rollout."""

    def from_parts(
        self,
        *,
        system_parts: dict[str, str] | None = None,
        user_parts: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AssembledContext:
        blocks: list[PromptBlock] = []
        for name, content in (system_parts or {}).items():
            if content and content.strip():
                blocks.append(PromptBlock(name=name, content=content, role=PromptBlockRole.SYSTEM))
        for name, content in (user_parts or {}).items():
            if content and content.strip():
                blocks.append(PromptBlock(name=name, content=content, role=PromptBlockRole.USER))
        return AssembledContext(blocks=blocks, metadata=dict(metadata or {}))
