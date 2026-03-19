from __future__ import annotations

from src.chat.utils.utils import process_llm_response, split_into_sentences_w_remove_punctuation
from src.config.config import global_config


def test_splitter_keeps_spaced_chinese_style_in_single_sentence():
    text = "晚 上 好 你 这 一 天 打 卡 挺 勤 快 啊"

    assert split_into_sentences_w_remove_punctuation(text) == [text]


def test_splitter_still_splits_spaced_style_on_punctuation(monkeypatch):
    monkeypatch.setattr("src.chat.utils.utils.random.random", lambda: 1.0)
    text = "晚 上 好。你 又 来 了"

    assert split_into_sentences_w_remove_punctuation(text) == ["晚 上 好", "你 又 来 了"]


def test_splitter_uses_newline_as_explicit_message_boundary(monkeypatch):
    monkeypatch.setattr("src.chat.utils.utils.random.random", lambda: 1.0)
    text = "晚上好\n这会儿才冒头啊"

    assert split_into_sentences_w_remove_punctuation(text) == ["晚上好", "这会儿才冒头啊"]


def test_process_llm_response_does_not_overflow_on_spaced_chinese_style(monkeypatch):
    class FakeTypoGenerator:
        def __init__(self, **kwargs):
            pass

        def create_typo_sentence(self, sentence: str):
            return sentence, ""

    monkeypatch.setattr("src.chat.utils.utils.ChineseTypoGenerator", FakeTypoGenerator)
    monkeypatch.setattr(global_config.response_post_process, "enable_response_post_process", True)
    monkeypatch.setattr(global_config.response_splitter, "enable", True)
    monkeypatch.setattr(global_config.response_splitter, "enable_kaomoji_protection", False)
    monkeypatch.setattr(global_config.response_splitter, "max_length", 100)
    monkeypatch.setattr(global_config.response_splitter, "max_sentence_num", 3)
    monkeypatch.setattr(global_config.response_splitter, "enable_overflow_return_all", False)
    monkeypatch.setattr(global_config.chinese_typo, "enable", False)

    text = "晚 上 好 你 这 一 天 打 卡 挺 勤 快 啊"
    result = process_llm_response(text, enable_splitter=True, enable_chinese_typo=False)

    assert result == [(text, 0.0)]
