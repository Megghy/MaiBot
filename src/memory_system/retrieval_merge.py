from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

INITIAL_INFO_RANK = 3000
LIVE_MEMORY_RESULT_RANK = 2200
LIVE_MODEL_RESULT_RANK = 2100
LIVE_CACHE_RESULT_RANK = 2000
CACHED_MEMORY_RESULT_RANK = 1000


@dataclass(slots=True)
class RankedMemoryBlock:
    block: str
    question: str = ""
    rank: int = 0
    order: int = 0
    origin: str = ""


def build_question_memory_block(
    *,
    question: str,
    block: str,
    raw_source: str,
    from_cache: bool,
    order: int,
) -> RankedMemoryBlock:
    if from_cache:
        rank = LIVE_CACHE_RESULT_RANK
    elif raw_source == "memory":
        rank = LIVE_MEMORY_RESULT_RANK
    elif raw_source == "model":
        rank = LIVE_MODEL_RESULT_RANK
    else:
        rank = LIVE_MODEL_RESULT_RANK - 50

    return RankedMemoryBlock(
        block=block,
        question=(question or "").strip(),
        rank=rank,
        order=order,
        origin="question_result",
    )


def build_cached_memory_block(*, question: str, block: str, order: int) -> RankedMemoryBlock | None:
    normalized_question = (question or "").strip()
    if not normalized_question:
        return None

    return RankedMemoryBlock(
        block=block,
        question=normalized_question,
        rank=CACHED_MEMORY_RESULT_RANK,
        order=order,
        origin="cached_memory",
    )


def merge_ranked_memory_blocks(
    *,
    initial_info: str,
    question_blocks: Sequence[RankedMemoryBlock],
    cached_blocks: Sequence[RankedMemoryBlock],
) -> list[str]:
    merged_by_question: dict[str, RankedMemoryBlock] = {}

    for block in [*question_blocks, *cached_blocks]:
        question = block.question.strip()
        if not question:
            continue

        current = merged_by_question.get(question)
        if current is None:
            merged_by_question[question] = block
            continue

        current_key = (current.rank, -current.order)
        candidate_key = (block.rank, -block.order)
        if candidate_key > current_key:
            merged_by_question[question] = block

    ordered_blocks = sorted(
        merged_by_question.values(),
        key=lambda item: (-item.rank, item.order, item.question),
    )

    prompt_blocks: list[str] = []
    initial_text = (initial_info or "").strip()
    if initial_text:
        prompt_blocks.append(initial_text)
    prompt_blocks.extend(block.block for block in ordered_blocks)
    return prompt_blocks


__all__ = [
    "RankedMemoryBlock",
    "build_cached_memory_block",
    "build_question_memory_block",
    "merge_ranked_memory_blocks",
]
