from __future__ import annotations

from dataclasses import dataclass, field

from src.runtime.agent.models import AgentProfile, ComposedPrompt, OutputMode, PromptBlock, PromptBlockRole


DEFAULT_BLOCK_ORDER = {
    "policy": 10,
    "identity": 20,
    "style": 30,
    "task": 40,
    "reply_target": 50,
    "chat_history": 60,
    "context": 70,
    "state": 80,
    "planner_hint": 90,
    "memory": 100,
    "tool_observations": 110,
    "mood_hint": 120,
    "chat_customization": 130,
    "output_contract": 200,
}


def build_output_contract(output_mode: OutputMode, *, allow_plain_text_fallback: bool) -> str:
    fallback_line = (
        "- 如果模型无法安全地调用工具, 可以退化为输出严格符合要求的纯文本结果"
        if allow_plain_text_fallback
        else "- 不允许使用纯文本替代工具调用或终止动作"
    )
    if output_mode == OutputMode.REPLY_TEXT:
        return "\n".join(
            [
                "输出要求:",
                "- 只输出最终回复内容",
                "- 不要输出解释、前后缀、角色标记、引号、括号、项目符号或多段分析",
                "- 回复保持自然、口语化、贴近当前聊天语境",
            ]
        )
    if output_mode == OutputMode.ACTION_SELECTION:
        return "\n".join(
            [
                "输出要求:",
                "- 优先使用 tool calls 执行动作选择",
                "- 不要输出动作说明、分析过程或额外闲聊",
                fallback_line,
            ]
        )
    if output_mode == OutputMode.QA_ANSWER:
        return "\n".join(
            [
                "输出要求:",
                "- 基于证据给出精简答案，不要添加无关前后缀",
                "- 如果当前流程定义了终止工具，必须通过终止工具提交最终结论",
                fallback_line,
            ]
        )
    if output_mode == OutputMode.TOOL_ONLY:
        return "\n".join(
            [
                "输出要求:",
                "- 仅通过 tool calls 执行操作",
                '- 如果确实不需要任何工具，只输出 "No tool needed"',
                fallback_line,
            ]
        )
    return ""


@dataclass(slots=True)
class PreparedPromptBlocks:
    blocks: list[PromptBlock] = field(default_factory=list)
    dropped_blocks: list[str] = field(default_factory=list)
    injected_blocks: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PromptComposer:
    block_order: dict[str, int] | None = None

    def prepare(self, blocks: list[PromptBlock], profile: AgentProfile) -> PreparedPromptBlocks:
        allowed_names = set(profile.enabled_context_blocks)
        prepared_blocks: list[PromptBlock] = []
        dropped_blocks: list[str] = []
        seen_names: set[str] = set()

        for block in blocks:
            if allowed_names and block.name not in allowed_names:
                dropped_blocks.append(block.name)
                continue
            prepared_blocks.append(block)
            seen_names.add(block.name)

        injected_blocks: list[str] = []
        should_inject_output_contract = "output_contract" in allowed_names if allowed_names else False
        if should_inject_output_contract and "output_contract" not in seen_names:
            output_contract = build_output_contract(
                profile.output_mode,
                allow_plain_text_fallback=profile.allow_plain_text_fallback,
            )
            if output_contract.strip():
                prepared_blocks.append(
                    PromptBlock(
                        name="output_contract",
                        content=output_contract,
                        role=PromptBlockRole.SYSTEM,
                        order=999,
                    )
                )
                injected_blocks.append("output_contract")

        return PreparedPromptBlocks(
            blocks=prepared_blocks,
            dropped_blocks=dropped_blocks,
            injected_blocks=injected_blocks,
        )

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
