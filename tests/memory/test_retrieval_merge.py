from __future__ import annotations

from src.memory_system.retrieval_merge import (
    build_cached_memory_block,
    build_question_memory_block,
    merge_ranked_memory_blocks,
)


def test_merge_ranked_memory_blocks_prefers_question_results_over_cached_duplicates():
    question_block = build_question_memory_block(
        question="她喜欢什么？",
        block="问题：她喜欢什么？\n答案：喜欢拉面\n来源：检索记忆",
        raw_source="memory",
        from_cache=False,
        order=0,
    )
    cached_duplicate = build_cached_memory_block(
        question="她喜欢什么？",
        block="问题：她喜欢什么？\n答案：喜欢米饭\n来源：历史缓存",
        order=0,
    )
    cached_unique = build_cached_memory_block(
        question="她最近在忙什么？",
        block="问题：她最近在忙什么？\n答案：最近在准备考试\n来源：历史缓存",
        order=1,
    )

    merged = merge_ranked_memory_blocks(
        initial_info="【概念检索结果】\n'术语A' 的含义为：解释A",
        question_blocks=[question_block],
        cached_blocks=[cached_duplicate, cached_unique],
    )

    assert merged == [
        "【概念检索结果】\n'术语A' 的含义为：解释A",
        "问题：她喜欢什么？\n答案：喜欢拉面\n来源：检索记忆",
        "问题：她最近在忙什么？\n答案：最近在准备考试\n来源：历史缓存",
    ]


def test_merge_ranked_memory_blocks_ranks_live_memory_over_live_model():
    model_block = build_question_memory_block(
        question="问题A",
        block="问题：问题A\n答案：模型答案\n来源：模型常识",
        raw_source="model",
        from_cache=False,
        order=0,
    )
    memory_block = build_question_memory_block(
        question="问题B",
        block="问题：问题B\n答案：检索答案\n来源：检索记忆",
        raw_source="memory",
        from_cache=False,
        order=1,
    )

    merged = merge_ranked_memory_blocks(
        initial_info="",
        question_blocks=[model_block, memory_block],
        cached_blocks=[],
    )

    assert merged == [
        "问题：问题B\n答案：检索答案\n来源：检索记忆",
        "问题：问题A\n答案：模型答案\n来源：模型常识",
    ]
