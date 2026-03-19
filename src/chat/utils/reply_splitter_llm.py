from __future__ import annotations

import re

from src.common.logger import get_logger
from src.config.config import global_config, model_config
from src.llm_models.exceptions import EmptyResponseException, NetworkConnectionError, RespNotOkException
from src.llm_models.utils_model import LLMRequest


logger = get_logger("reply_splitter")

SPLITTER_SYSTEM_PROMPT = """
你是聊天节奏整理器，不是改写器。

你的唯一任务：
- 判断这句回复是否更适合一条发完，还是拆成 2 到 3 条连续发送
- 你只能插入换行符 `\\n`

硬性规则：
- 禁止增删改任何其他字符
- 禁止补标点
- 禁止改空格、语气词、大小写、表情写法
- 如果一条发完更自然，就原样返回
- 只有明显像“先说一句，再补一句”时才换行
- 不要为了工整而强行分段
- 输出只能是处理后的文本本身
""".strip()


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _strip_outer_blank_lines(text: str) -> str:
    return _normalize_newlines(text).strip("\n")


def _sanitize_candidate_text(text: str) -> str:
    return re.sub(r"[ \t]*\n[ \t]*", "\n", _strip_outer_blank_lines(text))


def _should_use_llm_splitter(text: str) -> bool:
    splitter_config = global_config.response_splitter
    return (
        splitter_config.enable
        and splitter_config.enable_llm_splitter
        and "\n" not in text
        and len(text) >= splitter_config.llm_splitter_min_length
    )


def _build_splitter_user_prompt(text: str, target_message: str) -> str:
    prompt_lines = []
    if target_message.strip():
        prompt_lines.extend(
            [
                "对方刚发的是:",
                target_message,
                "",
            ]
        )
    prompt_lines.extend(
        [
            "原始回复:",
            text,
            "",
            "请返回更自然的发送节奏版本。",
        ]
    )
    return "\n".join(prompt_lines)


def _is_valid_split_result(original_text: str, candidate_text: str) -> bool:
    if not candidate_text or candidate_text == original_text:
        return bool(candidate_text)

    lines = candidate_text.split("\n")
    if any(not line.strip() for line in lines):
        return False
    if len(lines) > global_config.response_splitter.max_sentence_num:
        return False

    return candidate_text.replace("\n", "") == original_text.replace("\n", "")


async def maybe_split_reply_with_llm(text: str, *, target_message: str = "") -> str:
    source_text = _normalize_newlines(text)
    normalized_text = _strip_outer_blank_lines(source_text)
    if not normalized_text:
        return source_text
    if not _should_use_llm_splitter(normalized_text):
        return source_text

    splitter_task = model_config.model_task_config.reply_splitter
    request = LLMRequest(model_set=splitter_task, request_type="reply.splitter")

    try:
        response, (_, used_model, tool_calls) = await request.generate_response_async(
            _build_splitter_user_prompt(normalized_text, target_message),
            system_prompt=SPLITTER_SYSTEM_PROMPT,
            max_tokens=min(96, splitter_task.max_tokens),
        )
    except (NetworkConnectionError, EmptyResponseException, RespNotOkException) as exc:
        logger.warning(f"小模型分段失败，回退原文: {exc}")
        return source_text
    except Exception as exc:
        logger.warning(f"小模型分段异常，回退原文: {exc}")
        return source_text

    if tool_calls:
        logger.warning("小模型分段意外返回 tool calls，回退原文")
        return source_text

    candidate_text = _sanitize_candidate_text(response)
    if not _is_valid_split_result(normalized_text, candidate_text):
        logger.debug("小模型分段结果未通过一致性校验，回退原文")
        return source_text

    logger.debug(f"小模型分段成功，使用模型: {used_model}")
    return candidate_text
