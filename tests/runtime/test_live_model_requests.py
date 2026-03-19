from __future__ import annotations

from src.config.api_ada_configs import TaskConfig
from src.config.config import model_config
from src.llm_models.utils_model import LLMRequest

import pytest


def _build_single_model_task(task_name: str, model_name: str) -> TaskConfig:
    base_task = model_config.model_task_config.get_task(task_name)
    return TaskConfig(
        model_list=[model_name],
        max_tokens=base_task.max_tokens,
        temperature=base_task.temperature,
        frequency_penalty=base_task.frequency_penalty,
        presence_penalty=base_task.presence_penalty,
    )


def _build_live_prompt(task_name: str) -> str:
    if task_name == "replyer":
        return "你正在群聊中聊天。有人说：晚上好麦麦。请只回复一条自然、简短的中文消息。"
    if task_name == "reply_splitter":
        return "原文：都这个点了食堂还有啥好吃的 要不点外卖凑合一下 或者便利店买点\n只允许插入换行，不要改任何其他字符。"
    if task_name == "planner":
        return "你是一个决策模型。请只回复 OK。"
    if task_name == "tool_use":
        return '如果你不需要工具，请只回复 "No tool needed"。'
    if task_name == "expression":
        return "请只写一句 8 到 16 个字的自然中文口语短句。"
    if task_name in {"lpmm_entity_extract", "lpmm_rdf_build", "lpmm_qa"}:
        return "请只回复 测试通过。"
    return "请只回复 测试通过。"


@pytest.mark.live_model
@pytest.mark.asyncio
async def test_live_model_request_against_configured_models(live_model_case):
    task_name, model_name = live_model_case
    request = LLMRequest(
        model_set=_build_single_model_task(task_name, model_name),
        request_type=f"live_test.{task_name}",
    )

    if task_name == "embedding":
        embedding, used_model = await request.get_embedding("MaiBot live model test")
        assert used_model == model_name
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        return

    response, (reasoning, used_model, tool_calls) = await request.generate_response_async(
        _build_live_prompt(task_name),
        max_tokens=min(128, _build_single_model_task(task_name, model_name).max_tokens),
    )

    assert used_model == model_name
    assert isinstance(reasoning, str)
    assert response.strip() or tool_calls
