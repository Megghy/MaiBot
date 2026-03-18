from __future__ import annotations

from dataclasses import dataclass

from src.runtime.agent.models import ComposedPrompt, PromptBlock, PromptBlockRole


DEFAULT_BLOCK_ORDER = {
    "policy": 10,
    "identity": 20,
    "task": 30,
    "context": 40,
    "memory": 50,
    "tool_observations": 60,
    "output_contract": 70,
}


@dataclass(slots=True)
class PromptComposer:
    block_order: dict[str, int] | None = None

    def compose(self, blocks: list[PromptBlock]) -> ComposedPrompt:
        order = self.block_order or DEFAULT_BLOCK_ORDER

        def sort_key(block: PromptBlock) -> tuple[int, int, str]:
            return order.get(block.name, 1000), block.order, block.name

        ordered_blocks = [block for block in sorted(blocks, key=sort_key) if block.content.strip()]
        system_blocks = [block.content.strip() for block in ordered_blocks if block.role == PromptBlockRole.SYSTEM]
        user_blocks = [block.content.strip() for block in ordered_blocks if block.role == PromptBlockRole.USER]
        return ComposedPrompt(
            system_prompt="\n\n".join(system_blocks).strip(),
            user_prompt="\n\n".join(user_blocks).strip(),
            ordered_blocks=ordered_blocks,
        )
