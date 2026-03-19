from __future__ import annotations

import pytest

from src.config.config import global_config
from src.plugin_system.apis.generator_api import process_human_text
from src.chat.utils.reply_splitter_llm import maybe_split_reply_with_llm


@pytest.mark.asyncio
async def test_llm_splitter_skips_short_text(monkeypatch):
    monkeypatch.setattr(global_config.response_splitter, "enable", True)
    monkeypatch.setattr(global_config.response_splitter, "enable_llm_splitter", True)
    monkeypatch.setattr(global_config.response_splitter, "llm_splitter_min_length", 20)

    class FailRequest:
        def __init__(self, *args, **kwargs):
            raise AssertionError("short text should not call LLM splitter")

    monkeypatch.setattr("src.chat.utils.reply_splitter_llm.LLMRequest", FailRequest)

    result = await maybe_split_reply_with_llm("行 看到你了", target_message="打卡")

    assert result == "行 看到你了"


@pytest.mark.asyncio
async def test_llm_splitter_accepts_newline_only_output(monkeypatch):
    monkeypatch.setattr(global_config.response_splitter, "enable", True)
    monkeypatch.setattr(global_config.response_splitter, "enable_llm_splitter", True)
    monkeypatch.setattr(global_config.response_splitter, "llm_splitter_min_length", 10)
    monkeypatch.setattr(global_config.response_splitter, "max_sentence_num", 3)

    class FakeRequest:
        def __init__(self, *args, **kwargs):
            pass

        async def generate_response_async(self, *args, **kwargs):
            return "都这个点了 食堂还有啥好吃的\n要不点外卖凑合一下", ("", "fake-model", None)

    monkeypatch.setattr("src.chat.utils.reply_splitter_llm.LLMRequest", FakeRequest)

    result = await maybe_split_reply_with_llm(
        "都这个点了 食堂还有啥好吃的要不点外卖凑合一下",
        target_message="晚饭吃什么好",
    )

    assert result == "都这个点了 食堂还有啥好吃的\n要不点外卖凑合一下"


@pytest.mark.asyncio
async def test_llm_splitter_rejects_rewritten_output(monkeypatch):
    monkeypatch.setattr(global_config.response_splitter, "enable", True)
    monkeypatch.setattr(global_config.response_splitter, "enable_llm_splitter", True)
    monkeypatch.setattr(global_config.response_splitter, "llm_splitter_min_length", 10)
    monkeypatch.setattr(global_config.response_splitter, "max_sentence_num", 3)

    class FakeRequest:
        def __init__(self, *args, **kwargs):
            pass

        async def generate_response_async(self, *args, **kwargs):
            return "都这个点了 食堂还有啥好吃的\n不如点外卖凑合一下", ("", "fake-model", None)

    monkeypatch.setattr("src.chat.utils.reply_splitter_llm.LLMRequest", FakeRequest)

    original = "都这个点了 食堂还有啥好吃的要不点外卖凑合一下"
    result = await maybe_split_reply_with_llm(original, target_message="晚饭吃什么好")

    assert result == original


@pytest.mark.asyncio
async def test_llm_splitter_accepts_spacing_artifacts_around_newlines(monkeypatch):
    monkeypatch.setattr(global_config.response_splitter, "enable", True)
    monkeypatch.setattr(global_config.response_splitter, "enable_llm_splitter", True)
    monkeypatch.setattr(global_config.response_splitter, "llm_splitter_min_length", 10)
    monkeypatch.setattr(global_config.response_splitter, "max_sentence_num", 3)

    class FakeRequest:
        def __init__(self, *args, **kwargs):
            pass

        async def generate_response_async(self, *args, **kwargs):
            return "都这个点了 食堂还有啥好吃的  \n要不点外卖凑合一下", ("", "fake-model", None)

    monkeypatch.setattr("src.chat.utils.reply_splitter_llm.LLMRequest", FakeRequest)

    result = await maybe_split_reply_with_llm(
        "都这个点了 食堂还有啥好吃的要不点外卖凑合一下",
        target_message="晚饭吃什么好",
    )

    assert result == "都这个点了 食堂还有啥好吃的\n要不点外卖凑合一下"


@pytest.mark.asyncio
async def test_process_human_text_uses_llm_splitter_result(monkeypatch):
    async def fake_splitter(content: str, *, target_message: str = "") -> str:
        assert content == "晚上好呀 这会儿才冒泡吃晚饭没"
        assert target_message == "晚上好麦麦"
        return "晚上好呀 这会儿才冒泡\n吃晚饭没"

    def fake_process_llm_response(content: str, enable_splitter: bool = True, enable_chinese_typo: bool = True):
        assert content == "晚上好呀 这会儿才冒泡\n吃晚饭没"
        return [("晚上好呀 这会儿才冒泡", 0.0), ("吃晚饭没", 1.5)]

    monkeypatch.setattr("src.plugin_system.apis.generator_api.maybe_split_reply_with_llm", fake_splitter)
    monkeypatch.setattr("src.plugin_system.apis.generator_api.process_llm_response", fake_process_llm_response)

    reply_set = await process_human_text(
        "晚上好呀 这会儿才冒泡吃晚饭没",
        enable_splitter=True,
        enable_chinese_typo=False,
        target_message="晚上好麦麦",
    )

    assert reply_set is not None
    assert len(reply_set.reply_data) == 2
    assert reply_set.reply_data[0].content == "晚上好呀 这会儿才冒泡"
    assert reply_set.reply_data[1].content == "吃晚饭没"
