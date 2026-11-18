"""基于 LLM 的回复断句。"""

from typing import List, Tuple

import json

from src.common.logger import get_logger
from src.services.llm_service import LLMServiceClient

logger = get_logger("llm_sentence_splitter")

_NO_SPLIT_MARKER = "no_split"
_SINGLE_LINE_CHAR_LIMIT = 15


_SPLITTER_PROMPT = """你是一个即时通讯（QQ群聊）消息断句助手。请判断下面这条即将发送的回复是否适合拆成多条短消息。
要求：
1. 只能返回 JSON 字符串数组，例如 ["第一句", "第二句"]，或者返回 "no_split"，不要 Markdown、解释或代码块。
2. 必须完整保留原文的每一个文字和标点符号（包括空格、逗号、句号、问号、感叹号，绝对不要删改或增补任何字词或标点），只能在原文已有位置切分，程序会自动处理段尾逗号。
3. 不要把括号、引号、颜文字、英文单词、数字或代码从中间拆开。
4. 15字以内的单行短句一律保持一行，返回 "no_split"。只有当原文包含换行符，或者超过15字且确实包含多个独立意群/明显情绪转折时，才按自然停顿拆成 2~3 条气泡消息；若是句意连贯、仅用空格或单个逗号作轻度停顿的短吐槽/单句（约16~22字以内），优先返回 "no_split" 保持一行。完整代码、公式、表格、结构化清单也返回 "no_split"。

示例：
原文：大乱炖
五秒火速参悟
输出：["大乱炖", "五秒火速参悟"]

原文：算盘珠子崩脸上了 线上嫌菜 线下另算
输出：["算盘珠子崩脸上了", "线上嫌菜", "线下另算"]

原文：一眼淀粉肠 你心理暗示啥呢
输出："no_split"

原文：确实挺离谱的
输出："no_split"

原文：
"""


def _normalize_json_payload(payload: str) -> str:
    """去除模型偶尔添加的 Markdown JSON 代码块包装。"""
    normalized = payload.strip()
    if normalized.startswith("```json") and normalized.endswith("```"):
        return normalized[7:-3].strip()
    if normalized.startswith("```") and normalized.endswith("```"):
        return normalized[3:-3].strip()
    return normalized


def _is_no_split_marker(payload: str) -> bool:
    """判断模型返回的是否为「不拆分」标记，兼容是否被写成 JSON 字符串。"""
    return payload.strip().strip('"').strip() == _NO_SPLIT_MARKER


_PUNCTUATION_TABLE = {
    **{code: code - 0xFEE0 for code in range(0xFF01, 0xFF5F)},  # 全角 ASCII（含，。！？～等）
    0x3000: 0x20,
    0x3002: ord("."),
    0x3001: ord(","),
    0x2018: ord("'"),
    0x2019: ord("'"),
    0x201C: ord('"'),
    0x201D: ord('"'),
    0x3010: ord("["),
    0x3011: ord("]"),
    0x300A: ord("<"),
    0x300B: ord(">"),
}


def _normalize_char(char: str) -> str:
    """将单个字符按标点表归一化，用于对齐比较。"""
    return char.translate(_PUNCTUATION_TABLE)


def _strip_whitespace(text: str) -> str:
    """去除全部空白字符，并把中英文标点按一一对应归一，用于宽松比较。"""
    return "".join(text.split()).translate(_PUNCTUATION_TABLE)


def _align_sentences_to_original(text: str, sentences: List[str]) -> List[str]:
    """逐字校验切分结果保留了原文，并返回精确对应的原文片段。

    当模型偶尔在段尾切分处漏掉原文中的逗号（`，`/`,`）时，
    在不改动任何其他字符的前提下把原文逗号归并到前一段末尾。
    """
    normalized_segments = [_strip_whitespace(sentence) for sentence in sentences]
    normalized_segments = [segment for segment in normalized_segments if segment]
    if not normalized_segments:
        raise ValueError("LLM 断句结果不包含有效内容")

    result: List[str] = []
    pos = 0
    text_len = len(text)

    for seg_idx, segment_chars in enumerate(normalized_segments):
        while pos < text_len and text[pos].isspace():
            pos += 1
        seg_start = pos

        for expected_char in segment_chars:
            while pos < text_len and text[pos].isspace():
                pos += 1
            if pos >= text_len or _normalize_char(text[pos]) != expected_char:
                actual_snippet = text[max(0, pos - 10) : min(text_len, pos + 15)]
                raise ValueError(
                    "LLM 断句结果未完整保留原文（忽略空白后）："
                    f"原文位置={pos}，期望={expected_char!r}，原文片段={actual_snippet!r}"
                )
            pos += 1

        sentence_end = pos
        is_last = seg_idx == len(normalized_segments) - 1

        if is_last:
            # 末尾若仅因模型漏掉收尾逗号，一并吸收；其他非空白剩余视为篡改原文
            while pos < text_len and text[pos].isspace():
                pos += 1
            if pos < text_len and text[pos] in "，,":
                pos += 1
                sentence_end = pos
                while pos < text_len and text[pos].isspace():
                    pos += 1
            if pos < text_len:
                raise ValueError(
                    "LLM 断句结果未完整保留原文结尾："
                    f"未匹配剩余={text[pos : pos + 25]!r}"
                )
            result.append(text[seg_start:sentence_end])
            break

        # 若模型在断句边界漏写了原文里的分隔逗号，把该逗号归属到当前段末尾
        next_starts_with_comma = normalized_segments[seg_idx + 1].startswith(",")
        if not next_starts_with_comma:
            probe = pos
            while probe < text_len and text[probe].isspace():
                probe += 1
            if probe < text_len and text[probe] in "，,":
                pos = probe + 1
                sentence_end = pos

        trailing_end = pos
        while trailing_end < text_len and text[trailing_end].isspace():
            trailing_end += 1

        # 段尾逗号后的空白保留供分隔符还原；其他句间空白在逐条发送时无意义，直接丢弃
        slice_end = trailing_end if text[sentence_end - 1] in "，," else sentence_end
        result.append(text[seg_start:slice_end])
        pos = trailing_end

    return result


def _extract_trailing_comma(sentence: str) -> Tuple[str, str]:
    """将段尾逗号移入分隔符，逐条发送时省略，压缩合并时恢复。"""
    content = sentence.rstrip("，, \t\r\n")
    separator = sentence[len(content) :]
    if content.strip() and ("，" in separator or "," in separator):
        return content, separator
    return sentence, ""


async def split_text_with_llm(text: str) -> List[Tuple[str, str]]:
    """调用 LLM 断句，并返回兼容规则断句器的句子元组。"""
    normalized = text.strip()
    if not normalized:
        return []

    if "\n" not in normalized and len("".join(normalized.split())) <= _SINGLE_LINE_CHAR_LIMIT:
        return [_extract_trailing_comma(normalized)]

    client = LLMServiceClient(task_name="fast_model", request_type="response.splitter")
    result = await client.generate_response(_SPLITTER_PROMPT + normalized)
    payload = _normalize_json_payload(result.response)

    # 模型判断不需要拆分时只返回标记，此时作为单独一段，并同样处理段尾逗号
    if _is_no_split_marker(payload):
        return [_extract_trailing_comma(normalized)]

    try:
        sentences = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError("LLM 断句结果不是合法 JSON 数组") from exc

    if isinstance(sentences, str):
        raise ValueError("LLM 断句结果不是合法 JSON 数组")

    if not isinstance(sentences, list) or not sentences or not all(isinstance(item, str) for item in sentences):
        raise ValueError("LLM 断句结果必须是非空字符串数组")
    if sentences == [_NO_SPLIT_MARKER]:
        return [_extract_trailing_comma(normalized)]

    # 逐字对齐切分片段与原文（兼容边界漏逗号），并还原为原文切片再分离段尾逗号。
    original_sentences = _align_sentences_to_original(normalized, sentences)
    return [_extract_trailing_comma(sentence) for sentence in original_sentences]
