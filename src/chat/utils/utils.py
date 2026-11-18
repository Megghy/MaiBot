from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple

import asyncio
import random
import re
import time

from src.chat.message_receive.chat_manager import chat_manager as _chat_manager
from src.chat.message_receive.message import SessionMessage
from src.common.data_models.message_component_data_model import AtComponent
from src.common.logger import get_logger
from src.config.config import global_config
from src.person_info.person_info import Person
from src.services.bot_account_service import get_bot_accounts
from src.services.embedding_service import EmbeddingServiceClient

from .llm_sentence_splitter import split_text_with_llm
from .typo_generator import ChineseTypoGenerator

if TYPE_CHECKING:
    from src.common.data_models.chat_target_info_data_model import ChatTargetInfo

logger = get_logger("chat_utils")
_warned_unconfigured_platforms: set[str] = set()


@dataclass(frozen=True)
class ProcessedResponseSegment:
    """回复后处理产生的单条消息及其发送提示。"""

    text: str
    quote_previous: bool = False
    # 原始文本中紧跟本段的分隔符；多条消息被压缩合并为一条时，用它恢复句间的停顿
    separator: str = ""
    # 纠正内容始终独占一条消息，不依赖是否启用引用。
    is_correction: bool = False


def is_english_letter(char: str) -> bool:
    """检查字符是否为英文字母（忽略大小写）"""
    return "a" <= char.lower() <= "z"


def parse_platform_accounts(platforms: list[str]) -> dict[str, str]:
    """解析 platforms 列表，返回平台到账号的映射

    Args:
        platforms: 格式为 ["platform:account"] 的列表，如 ["tg:123456789", "wx:wxid123"]

    Returns:
        字典，键为平台名，值为账号
    """
    result: dict[str, str] = {}
    for platform_entry in platforms:
        if ":" in platform_entry:
            platform_name, account = platform_entry.split(":", 1)
            normalized_platform = platform_name.lower().strip()
            account_str = account.strip()
            if normalized_platform and account_str:
                result[normalized_platform] = account_str
    return result


def _get_configured_qq_account() -> str:
    qq_account = global_config.bot.qq_account.strip()
    if qq_account in {"", "0"}:
        return ""
    return qq_account


def get_bot_account(platform: str, preferred_account_id: Optional[str] = None) -> str:
    """解析单个 Bot 账号；多账号且无会话归属时拒绝猜测。"""

    preferred = str(preferred_account_id or "").strip()
    if preferred:
        return preferred
    accounts = get_bot_accounts(platform)
    if len(accounts) == 1:
        return next(iter(accounts))
    if len(accounts) > 1:
        logger.error(f"平台 {platform} 存在多个 Bot 账号，缺少聊天流 account_id，无法选择发送身份")
    return ""


def get_configured_bot_accounts() -> dict[str, str]:
    """读取 legacy 发送链使用的备用配置账号，不包含数据库身份。"""

    bot_accounts: dict[str, str] = {}
    qq_account = _get_configured_qq_account()
    if qq_account:
        primary_platform = str(global_config.bot.platform or "qq").strip().lower()
        bot_accounts[primary_platform] = qq_account

    platforms_list = global_config.bot.platforms
    platform_accounts = parse_platform_accounts(platforms_list)

    for platform_name, account in platform_accounts.items():
        if platform_name in {"qq", "webui", "internal"}:
            continue
        bot_accounts[platform_name] = account

    return bot_accounts


def is_bot_self(platform: str, user_id: str) -> bool:
    """判断给定的平台和用户ID是否是机器人自己

    这个函数统一处理所有平台（包括 QQ、Telegram、WebUI 等）的机器人识别逻辑。

    Args:
        platform: 消息平台（如 "qq", "telegram", "webui" 等）
        user_id: 用户ID

    Returns:
        bool: 如果是机器人自己则返回 True，否则返回 False
    """
    normalized_platform = str(platform or "").strip().lower()
    if not normalized_platform or not user_id:
        return False

    # 将 user_id 转为字符串进行比较
    user_id_str = str(user_id).strip()
    if not user_id_str:
        return False

    bot_accounts = get_bot_accounts(normalized_platform)
    if bot_accounts:
        return user_id_str in bot_accounts

    if normalized_platform not in _warned_unconfigured_platforms:
        _warned_unconfigured_platforms.add(normalized_platform)
        logger.warning(f"平台 {normalized_platform} 未配置机器人账号，无法判断用户 {user_id_str} 是否为机器人自己")
    return False


def _has_at_component_targeting_bot(message: SessionMessage, platform: str) -> bool:
    """检查消息中的结构化 @ 组件是否直接指向当前 bot。"""

    raw_message = getattr(message, "raw_message", None)
    for component in getattr(raw_message, "components", []) or []:
        if isinstance(component, AtComponent) and is_bot_self(platform, component.target_user_id):
            return True
    return False


def is_mentioned_bot_in_message(message: SessionMessage) -> tuple[bool, bool, float]:
    """检查消息是否提到了机器人（统一多平台实现）"""
    text = message.processed_plain_text or ""
    platform = str(message.platform or "").strip().lower()

    current_accounts = get_bot_accounts(platform)

    nickname = global_config.bot.nickname
    alias_names = global_config.bot.alias_names
    keywords = [nickname] + alias_names

    reply_probability = 0.0
    is_at = False
    is_mentioned = False

    # 1) 直接的 additional_config 标记
    add_cfg = getattr(message.message_info, "additional_config", None) or {}
    if isinstance(add_cfg, dict):
        if add_cfg.get("at_bot") or add_cfg.get("is_mentioned"):
            is_mentioned = True
            if add_cfg.get("at_bot"):
                is_at = True
            # 当提供数值型 is_mentioned 时，当作概率提升；布尔提及标记只负责标记命中。
            raw_mention_boost = add_cfg.get("is_mentioned")
            if raw_mention_boost not in (None, "") and not isinstance(raw_mention_boost, bool):
                reply_probability = float(raw_mention_boost)

    # 2) 已经在上游设置过的 message.is_at / message.is_mentioned
    if getattr(message, "is_at", False):
        is_at = True
        is_mentioned = True
    if getattr(message, "is_mentioned", False):
        is_mentioned = True

    # 3) 扫描分段：是否包含 mention_bot（适配器插入）
    def _has_mention_bot(seg) -> bool:
        try:
            if seg is None:
                return False
            if getattr(seg, "type", None) == "mention_bot":
                return True
            if getattr(seg, "type", None) == "seglist":
                for s in getattr(seg, "data", []) or []:
                    if _has_mention_bot(s):
                        return True
            return False
        except Exception:
            return False

    if _has_mention_bot(getattr(message, "message_segment", None)):
        is_at = True
        is_mentioned = True

    # 4) 结构化 @ 组件检测。处理后的文本可能只剩群名片，不能依赖文本里的显示名判断。
    if not is_at and _has_at_component_targeting_bot(message, platform):
        is_at = True
        is_mentioned = True

    # 5) 统一的 @ 检测逻辑
    if current_accounts and not is_at and not is_mentioned:
        for current_account in current_accounts:
            pattern = (
                rf"@<(.+?):{re.escape(current_account)}>"
                if platform == "qq"
                else rf"@{re.escape(current_account)}(\b|$)"
            )
            if re.search(pattern, text, flags=re.IGNORECASE):
                is_at = True
                is_mentioned = True
                break

    # 6) 统一的回复检测逻辑
    if not is_mentioned:
        # 通用回复格式：包含 "(你)" 或 "（你）"
        if re.search(r"\[回复 .*?\(你\)：", text) or re.search(r"\[回复 .*?（你）：", text):
            is_mentioned = True
        # ID 形式的回复检测
        elif current_accounts:
            for current_account in current_accounts:
                if re.search(rf"\[回复 (.+?)\({re.escape(current_account)}\)：(.+?)\]，说：", text):
                    is_mentioned = True
                    break
                if re.search(
                    rf"\[回复<(.+?)(?=:{re.escape(current_account)}>)\:{re.escape(current_account)}>：(.+?)\]，说：",
                    text,
                ):
                    is_mentioned = True
                    break

    # 7) 名称/别名 提及（去除 @/回复标记后再匹配）
    if not is_mentioned and keywords:
        msg_content = text
        # 去除各种 @ 与 回复标记，避免误判
        msg_content = re.sub(r"@(.+?)（(\d+)）", "", msg_content)
        msg_content = re.sub(r"@<(.+?)(?=:(\d+))\:(\d+)>", "", msg_content)
        msg_content = re.sub(r"\[回复 (.+?)\(((\d+)|未知id|你)\)：(.+?)\]，说：", "", msg_content)
        msg_content = re.sub(r"\[回复<(.+?)(?=:(\d+))\:(\d+)>：(.+?)\]，说：", "", msg_content)
        for kw in keywords:
            if kw and kw in msg_content:
                is_mentioned = True
                break

    # 8) 概率设置
    reply_timing_config = global_config.chat.reply_timing
    if is_at and reply_timing_config.inevitable_at_reply:
        reply_probability = 1.0
        logger.debug("被@，回复概率设置为100%")
    elif is_mentioned and reply_timing_config.mentioned_bot_reply:
        reply_probability = max(reply_probability, 1.0)
        logger.debug("被提及，回复概率设置为100%")

    return is_mentioned, is_at, reply_probability


async def get_embedding(text: str, request_type: str = "embedding") -> Optional[List[float]]:
    """获取文本的嵌入向量。

    Args:
        text: 待编码的文本内容。
        request_type: 当前请求的业务类型标识。

    Returns:
        Optional[List[float]]: 成功时返回嵌入向量，失败时返回 `None`。
    """
    embedding_client = EmbeddingServiceClient(task_name="embedding", request_type=request_type)
    try:
        embedding_result = await embedding_client.embed_text(text)
        embedding = embedding_result.embedding
    except Exception as e:
        logger.error(f"获取embedding失败: {str(e)}")
        embedding = None
    return embedding


_OPEN_TO_CLOSE_QUOTES = {
    '"': '"',
    "'": "'",
    "“": "”",
    "‘": "’",
    "「": "」",
    "『": "』",
}
_CLOSING_QUOTE_CHARS = {"”", "’", "」", "』"}
_TERMINAL_PUNCT_CHARS = {"？", "?", "！", "!"}
_DISCARD_SEPARATORS = {"，", ",", " ", "。", ";", "；", "\n"}
_CHINESE_BRACKET_PATTERN = re.compile(r"[(\[（](?=.*[一-鿿]).*?[)\]）]")
_KAOMOJI_PLACEHOLDER_PATTERN = re.compile(r"__KAOMOJI_\d+__")


def _compute_inside_quote_mask(text: str) -> list[bool]:
    """仅对真正成对闭合的同段引号区间标记为 True，避免未闭合引号或英文撇号锁死后续分段。"""
    inside_quote = [False] * len(text)
    quote_stack: list[tuple[int, str]] = []

    for idx, ch in enumerate(text):
        if ch == "\n":
            # 聊天文本的引用不跨换行，换行即重置未闭合引号栈防止污染后续行
            quote_stack.clear()
            continue

        if ch == "'" and idx > 0:
            prev_char = text[idx - 1]
            next_char = text[idx + 1] if idx + 1 < len(text) else ""
            # 跳过英文缩写（it's / don't）和未开引号的英语所有格（girls'）
            if prev_char.isascii() and prev_char.isalnum() and next_char.isascii() and next_char.isalnum():
                continue
            if prev_char.lower() == "s" and not any(expected == "'" for _, expected in quote_stack):
                continue

        if quote_stack and ch == quote_stack[-1][1]:
            open_idx, _ = quote_stack.pop()
            for inner_idx in range(open_idx + 1, idx):
                inside_quote[inner_idx] = True
        elif ch in _CLOSING_QUOTE_CHARS:
            match_pos = next(
                (pos for pos in range(len(quote_stack) - 1, -1, -1) if quote_stack[pos][1] == ch),
                None,
            )
            if match_pos is not None:
                open_idx, _ = quote_stack[match_pos]
                del quote_stack[match_pos:]
                for inner_idx in range(open_idx + 1, idx):
                    inside_quote[inner_idx] = True
        elif ch in _OPEN_TO_CLOSE_QUOTES:
            quote_stack.append((idx, _OPEN_TO_CLOSE_QUOTES[ch]))

    return inside_quote


def _is_inside_url(text: str, index: int) -> bool:
    """判断指定位置是否处于 http(s) URL 内部，避免误切查询参数标点。"""
    if text[index].isspace():
        return False
    token_start = index
    while token_start > 0 and not text[token_start - 1].isspace():
        token_start -= 1
    token_prefix = text[token_start : index + 1].lower()
    return token_prefix.startswith(("http://", "https://"))


def _can_split_on_terminal_punct(text: str, index: int) -> bool:
    """判断句末感叹号/问号后是否适合作为切分边界。"""
    if _is_inside_url(text, index):
        return False
    char = text[index]
    if char in {"?", "!"} and index + 1 < len(text):
        next_char = text[index + 1]
        if next_char.isascii() and next_char.isalnum():
            return False
    return True


def _can_split_on_discard_separator(text: str, index: int) -> bool:
    """判断逗号/句号/分号/空格/换行是否可作为切分点。"""
    char = text[index]
    if char == "\n":
        return True
    if _is_inside_url(text, index):
        return False
    if index > 0 and text[index - 1] in {":", "："}:
        return False
    if index < len(text) - 1 and text[index + 1] in {":", "："}:
        return False
    if char == " " and 0 < index < len(text) - 1:
        prev_char = text[index - 1]
        next_char = text[index + 1]
        if prev_char in {"-", "—"} or next_char in {"-", "—"}:
            return False
        prev_is_alnum = prev_char.isdigit() or is_english_letter(prev_char)
        next_is_alnum = next_char.isdigit() or is_english_letter(next_char)
        if prev_is_alnum and next_is_alnum:
            return False
    return True


def _collapse_spaced_single_chars(segments: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """把连续 3 个及以上由单个空格隔开的单字拼回整句（兼容刻意加空格风格）。"""
    collapsed: list[tuple[str, str]] = []
    idx = 0
    total = len(segments)
    while idx < total:
        run_end = idx
        while run_end < total:
            content, sep = segments[run_end]
            if len(content.strip()) != 1 or not content.strip():
                break
            if sep != " " and run_end > idx:
                # 当前单字带非空格收尾标点时结束本次连续单字序列，并将其纳入整句
                run_end += 1
                break
            if sep != " ":
                break
            run_end += 1

        if run_end - idx >= 3:
            merged_content = "".join(
                content + (sep if pos < run_end - idx - 1 else "")
                for pos, (content, sep) in enumerate(segments[idx:run_end])
            )
            collapsed.append((merged_content, segments[run_end - 1][1]))
            idx = run_end
        else:
            collapsed.append(segments[idx])
            idx += 1
    return collapsed


def _effective_segment_length(content: str) -> int:
    """将颜文字占位符折算为实际视觉长度，避免占位符虚高影响合并阈值。"""
    return len(_KAOMOJI_PLACEHOLDER_PATTERN.sub("XX", content.strip()))


def _compute_merge_probability(
    left_content: str,
    left_sep: str,
    right_content: str,
    total_segments: int,
) -> float:
    """依据相邻两段自身长度与分隔符计算随机合并概率，避免短回复被高概率黏成一条。"""
    if left_sep == "\n":
        return 0.0
    if left_content.rstrip().endswith(("？", "?", "！", "!")):
        return 0.0

    left_len = _effective_segment_length(left_content)
    right_len = _effective_segment_length(right_content)
    if left_len <= 2 or right_len <= 2:
        return 0.65
    if left_len + len(left_sep) + right_len > 28:
        return 0.05
    if left_sep in {"。", ";", "；"}:
        return 0.10
    if total_segments <= 3:
        return 0.15
    return 0.30


def _collect_raw_sentence_segments(text: str, inside_quote: list[bool]) -> list[tuple[str, str]]:
    """遍历文本收集基础 (句子, 分隔符) 元组。"""
    len_text = len(text)
    segments: list[tuple[str, str]] = []
    current_segment = ""

    i = 0
    while i < len_text:
        char = text[i]
        if not inside_quote[i] and char in _TERMINAL_PUNCT_CHARS and _can_split_on_terminal_punct(text, i):
            current_segment += char
            while i + 1 < len_text and text[i + 1] in _TERMINAL_PUNCT_CHARS and not inside_quote[i + 1]:
                i += 1
                current_segment += text[i]

            probe = i + 1
            while probe < len_text and text[probe].isspace():
                probe += 1
            if probe >= len_text:
                i += 1
                continue

            trailing_sep = ""
            if i + 1 < len_text and text[i + 1] == "\n":
                trailing_sep = "\n"
                i += 1
            else:
                space_start = i + 1
                space_end = space_start
                while space_end < len_text and text[space_end] in {" ", "\t"}:
                    space_end += 1
                if space_end > space_start:
                    trailing_sep = text[space_start:space_end]
                    i = space_end - 1

            segments.append((current_segment, trailing_sep))
            current_segment = ""
        elif char in _DISCARD_SEPARATORS:
            can_split = not inside_quote[i] and _can_split_on_discard_separator(text, i)
            if can_split:
                if current_segment:
                    segments.append((current_segment, char))
                elif char in {" ", "\n"}:
                    segments.append(("", char))
                current_segment = ""
            else:
                current_segment += char
        else:
            current_segment += char
        i += 1

    if current_segment:
        segments.append((current_segment, ""))
    return [(content, sep) for content, sep in segments if content or sep]


def _split_into_sentence_segments(text: str) -> list[tuple[str, str]]:
    """将文本分割成 (句子, 尾随分隔符) 元组，并按段长和句界控制合并概率。"""
    text = re.sub(r"\n\s*\n+", "\n", text)
    text = re.sub(r"\n\s*([，,。;；\s])", r"\n\1", text)
    text = re.sub(r"([，,。;；\s])\s*\n", r"\1\n", text)

    if len(text) < 3:
        if random.random() < 0.01:
            return [(char, "") for char in text]
        return [(text, "")]

    inside_quote = _compute_inside_quote_mask(text)
    segments = _collect_raw_sentence_segments(text, inside_quote)
    if not segments:
        return [(text, "")] if text else []

    segments = _collapse_spaced_single_chars(segments)
    total_segments = len(segments)

    merged_segments: list[tuple[str, str]] = []
    idx = 0
    while idx < total_segments:
        current_content, current_sep = segments[idx]
        if idx + 1 < total_segments and current_content:
            next_content, next_sep = segments[idx + 1]
            merge_prob = _compute_merge_probability(current_content, current_sep, next_content, total_segments)
            if merge_prob > 0.0 and random.random() < merge_prob:
                if next_content:
                    merged_segments.append((current_content + current_sep + next_content, next_sep))
                else:
                    merged_segments.append((current_content, next_sep))
                idx += 2
                continue
        merged_segments.append((current_content, current_sep))
        idx += 1

    sentence_segments = [(content, sep) for content, sep in merged_segments if content and content.strip()]
    sentence_segments = [
        (normalized_sentence, sep)
        for sentence, sep in sentence_segments
        if (normalized_sentence := re.sub(r"[^\S\r\n]*[\r\n]+[^\S\r\n]*", " ", sentence).strip())
    ]

    logger.debug(f"分割并合并后的句子: {[sentence for sentence, _sep in sentence_segments]}")
    return sentence_segments


def split_into_sentences_w_remove_punctuation(text: str) -> list[str]:
    """将文本分割成句子，并根据概率合并（不保留分隔符信息，仅返回句子内容）"""
    return [sentence for sentence, _separator in _split_into_sentence_segments(text)]


def _merge_processed_segments_to_max_count(
    segments: List[ProcessedResponseSegment],
    max_count: int,
) -> List[ProcessedResponseSegment]:
    """压缩回复段数量；纠正内容独占消息所需的边界优先于条数上限。"""

    if len(segments) <= max_count:
        return segments
    if max_count <= 0:
        return []

    segment_count = len(segments)
    group_starts = {0}
    for index, segment in enumerate(segments):
        if segment.is_correction or segment.quote_previous:
            # 同时保留纠正段前后的边界，避免它吞入后续正文；上限不足时也不能舍弃。
            group_starts.add(index)
            if index + 1 < segment_count:
                group_starts.add(index + 1)

    # 在保留纠正消息边界后，沿用原来的均匀分组策略填充其余可用分组。
    evenly_spaced_starts: List[int] = []
    start_index = 0
    for group_index in range(max_count):
        remaining_segments = segment_count - start_index
        remaining_groups = max_count - group_index
        group_size = (remaining_segments + remaining_groups - 1) // remaining_groups
        evenly_spaced_starts.append(start_index)
        start_index += group_size

    for candidate_start in evenly_spaced_starts:
        if len(group_starts) >= max_count:
            break
        group_starts.add(candidate_start)

    sorted_starts = sorted(group_starts)
    merged_segments: List[ProcessedResponseSegment] = []
    for group_index, group_start in enumerate(sorted_starts):
        group_end = sorted_starts[group_index + 1] if group_index + 1 < len(sorted_starts) else segment_count
        group = segments[group_start:group_end]
        # 组内多段被压缩为一条消息时，补回各段尾随的分隔符，避免多句硬拼接成无标点的一整坨
        merged_text_parts: List[str] = []
        for position, segment in enumerate(group):
            merged_text_parts.append(segment.text)
            if position < len(group) - 1:
                merged_text_parts.append(segment.separator)
        merged_segments.append(
            ProcessedResponseSegment(
                text="".join(merged_text_parts),
                quote_previous=group[0].quote_previous,
                separator=group[-1].separator,
                is_correction=group[0].is_correction,
            )
        )

    return merged_segments


def _get_random_default_reply() -> str:
    """获取随机默认回复"""
    default_replies = [
        f"{global_config.bot.nickname}不知道哦",
        f"{global_config.bot.nickname}不知道",
        "不知道哦",
        "不知道",
        "不晓得",
        "懒得说",
        "()",
    ]
    return random.choice(default_replies)


def _apply_response_post_processing(
    split_sentences: list[tuple[str, str]],
    enable_chinese_typo: bool,
) -> list[ProcessedResponseSegment]:
    """对已完成断句的文本执行错别字注入和纠正段处理。"""
    # 两种断句模式共用此入口：关闭功能时不构建生成器，启用时复用共享数据。
    typo_generator = None
    if global_config.chinese_typo.enable and enable_chinese_typo:
        typo_generator = ChineseTypoGenerator(
            error_rate=global_config.chinese_typo.error_rate,
            min_freq=global_config.chinese_typo.min_freq,
            tone_error_rate=global_config.chinese_typo.tone_error_rate,
            word_replace_rate=global_config.chinese_typo.word_replace_rate,
        )
    segments: list[ProcessedResponseSegment] = []
    for sentence, sentence_separator in split_sentences:
        if typo_generator is not None:
            typoed_text, typo_corrections = typo_generator.create_typo_sentence(sentence)
            if typo_corrections:
                if random.random() < 0.5:
                    quote_previous = (
                        global_config.chinese_typo.enable_correction_quote
                        and random.random() < global_config.chinese_typo.correction_quote_probability
                    )
                    segments.append(ProcessedResponseSegment(typoed_text, separator=sentence_separator))
                    segments.append(
                        ProcessedResponseSegment(typo_corrections, quote_previous=quote_previous, is_correction=True)
                    )
                else:
                    segments.append(ProcessedResponseSegment(sentence, separator=sentence_separator))
            else:
                segments.append(ProcessedResponseSegment(typoed_text, separator=sentence_separator))
        else:
            segments.append(ProcessedResponseSegment(sentence, separator=sentence_separator))
    return segments


def _prepare_response_text(
    text: str,
    *,
    protect_kaomoji_tokens: bool,
) -> tuple[str, dict[str, str], Optional[list[ProcessedResponseSegment]]]:
    """统一执行颜文字保护、中文括号旁白清理与长度上限拦截。"""
    if protect_kaomoji_tokens and global_config.response_splitter.enable_kaomoji_protection:
        protected_text, kaomoji_mapping = protect_kaomoji(text)
        logger.debug(f"保护颜文字后的文本: {protected_text}")
    else:
        protected_text = text
        kaomoji_mapping = {}

    cleaned_text = _CHINESE_BRACKET_PATTERN.sub("", protected_text)
    if cleaned_text == "":
        return "", kaomoji_mapping, [ProcessedResponseSegment("呃呃")]

    logger.debug(f"{text}去除括号处理后的文本: {cleaned_text}")

    max_length = global_config.response_splitter.max_length * 2
    if get_western_ratio(cleaned_text) < 0.1 and len(cleaned_text) > max_length:
        logger.warning(f"回复过长 ({len(cleaned_text)} 字符)，返回默认回复")
        return cleaned_text, kaomoji_mapping, [ProcessedResponseSegment(_get_random_default_reply())]

    return cleaned_text, kaomoji_mapping, None


def _apply_post_processing_and_limits(
    split_sentences: list[tuple[str, str]],
    enable_chinese_typo: bool,
    kaomoji_mapping: dict[str, str],
) -> list[ProcessedResponseSegment]:
    """执行错别字注入、句数超限处理、最大条数合并与颜文字恢复。"""
    segments = _apply_response_post_processing(split_sentences, enable_chinese_typo)
    max_sentence_num = global_config.response_splitter.max_sentence_num
    max_split_num = global_config.response_splitter.max_split_num

    if len(segments) > max_sentence_num:
        if global_config.response_splitter.enable_overflow_return_all:
            logger.warning(f"分割后消息数量过多 ({len(segments)} 条)，按最大分割条数合并发送")
        else:
            logger.warning(f"分割后消息数量过多 ({len(segments)} 条)，返回默认回复")
            return [ProcessedResponseSegment(_get_random_default_reply())]

    segments = _merge_processed_segments_to_max_count(segments, max_split_num)

    if kaomoji_mapping:
        recovered_sentences = recover_kaomoji([segment.text for segment in segments], kaomoji_mapping)
        segments = [
            ProcessedResponseSegment(
                text=recovered_text,
                quote_previous=segment.quote_previous,
                separator=segment.separator,
                is_correction=segment.is_correction,
            )
            for segment, recovered_text in zip(segments, recovered_sentences, strict=True)
        ]

    return segments


def process_llm_response_segments(
    text: str,
    enable_splitter: bool = True,
    enable_chinese_typo: bool = True,
) -> list[ProcessedResponseSegment]:
    """处理回复文本，并保留错别字纠正消息的引用提示。"""

    if not global_config.response_post_process.enable_response_post_process:
        return [ProcessedResponseSegment(text)]

    cleaned_text, kaomoji_mapping, early_return = _prepare_response_text(text, protect_kaomoji_tokens=True)
    if early_return is not None:
        return early_return

    if global_config.response_splitter.enable and enable_splitter:
        split_sentences = _split_into_sentence_segments(cleaned_text)
    else:
        split_sentences = [(cleaned_text, "")]

    return _apply_post_processing_and_limits(split_sentences, enable_chinese_typo, kaomoji_mapping)


async def process_llm_response_segments_async(
    text: str,
    enable_splitter: bool = True,
    enable_chinese_typo: bool = True,
) -> list[ProcessedResponseSegment]:
    """异步处理回复文本，支持使用 LLM 按语义断句，失败时回退规则断句。"""
    if global_config.response_splitter.mode != "llm":
        return await asyncio.to_thread(process_llm_response_segments, text, enable_splitter, enable_chinese_typo)

    if (
        not global_config.response_post_process.enable_response_post_process
        or not global_config.response_splitter.enable
        or not enable_splitter
    ):
        return await asyncio.to_thread(process_llm_response_segments, text, enable_splitter, enable_chinese_typo)

    cleaned_text, _mapping, early_return = _prepare_response_text(text, protect_kaomoji_tokens=False)
    if early_return is not None:
        return early_return

    try:
        split_sentences = await split_text_with_llm(cleaned_text)
    except Exception as exc:
        logger.warning(f"LLM 断句失败，回退规则断句：{exc}")
        return await asyncio.to_thread(process_llm_response_segments, text, enable_splitter, enable_chinese_typo)

    # 数据首次加载和错别字计算交给线程执行，避免占住聊天事件循环。
    return await asyncio.to_thread(_apply_post_processing_and_limits, split_sentences, enable_chinese_typo, {})


def process_llm_response(text: str, enable_splitter: bool = True, enable_chinese_typo: bool = True) -> list[str]:
    """处理回复文本，并返回兼容旧调用链的纯文本列表。"""

    return [
        segment.text
        for segment in process_llm_response_segments(
            text,
            enable_splitter=enable_splitter,
            enable_chinese_typo=enable_chinese_typo,
        )
    ]


def calculate_typing_time(
    input_string: str,
    # thinking_start_time: float,
    chinese_time: float = 0.3,
    english_time: float = 0.15,
    is_emoji: bool = False,
    is_image: bool = False,
) -> float:
    """
    计算输入字符串所需的时间，中文和英文字符有不同的输入时间
        input_string (str): 输入的字符串
        chinese_time (float): 中文字符的输入时间，默认为0.2秒
        english_time (float): 英文字符的输入时间，默认为0.1秒
        is_emoji (bool): 是否为emoji，默认为False
        is_image (bool): 是否为纯图片消息，默认为False

    特殊情况：
    - 如果只有一个中文字符，将使用3倍的中文输入时间
    - 在所有输入结束后，额外加上回车时间0.3秒
    - 如果is_emoji或is_image为True，将使用固定1秒的输入时间，再应用打字速度配置
    """
    # chinese_time *= 1 / typing_speed_multiplier
    # english_time *= 1 / typing_speed_multiplier
    # 计算中文字符数
    chinese_chars = sum("\u4e00" <= char <= "\u9fff" for char in input_string)

    # 如果只有一个中文字符，使用3倍时间
    if not (is_emoji or is_image) and chinese_chars == 1 and len(input_string.strip()) == 1:
        return chinese_time * 3 + 0.3  # 加上回车时间

    # 正常计算所有字符的输入时间
    total_time = 0.0
    for char in input_string:
        total_time += chinese_time if "\u4e00" <= char <= "\u9fff" else english_time
    if is_emoji or is_image:
        total_time = 1

    typing_speed = global_config.response_post_process.typing_speed
    if typing_speed <= 0:
        return 0
    total_time *= typing_speed

    # if time.time() - thinking_start_time > 10:
    #     total_time = 1

    # print(f"thinking_start_time:{thinking_start_time}")
    # print(f"nowtime:{time.time()}")
    # print(f"nowtime - thinking_start_time:{time.time() - thinking_start_time}")
    # print(f"{total_time}")

    return total_time  # 加上回车时间


def truncate_message(message: str, max_length=20) -> str:
    """截断消息，使其不超过指定长度"""
    return f"{message[:max_length]}..." if len(message) > max_length else message


def protect_kaomoji(sentence):
    """ "
    识别并保护句子中的颜文字（含括号与无括号），将其替换为占位符，
    并返回替换后的句子和占位符到颜文字的映射表。
    Args:
        sentence (str): 输入的原始句子
    Returns:
        tuple: (处理后的句子, {占位符: 颜文字})
    """
    kaomoji_pattern = re.compile(
        r"("
        r"[(\[（【]"  # 左括号
        r"[^()\[\]（）【】]*?"  # 非括号字符（惰性匹配）
        r"[^一-龥a-zA-Z0-9\s]"  # 非中文、非英文、非数字、非空格字符（必须包含至少一个）
        r"[^()\[\]（）【】]*?"  # 非括号字符（惰性匹配）
        r"[)\]）】"  # 右括号
        r"]"
        r")"
        r"|"
        r"([▼▽・ᴥω･﹏^><≧≦￣｀´∀ヮДд︿﹀へ｡ﾟ╥╯╰︶︹•⁄]{2,15})"
    )

    kaomoji_matches = kaomoji_pattern.findall(sentence)
    placeholder_to_kaomoji = {}

    for match in kaomoji_matches:
        kaomoji = match[0] or match[1]
        if kaomoji.startswith("[表情包") and kaomoji.endswith("]"):
            continue
        idx = len(placeholder_to_kaomoji)
        placeholder = f"__KAOMOJI_{idx}__"
        sentence = sentence.replace(kaomoji, placeholder, 1)
        placeholder_to_kaomoji[placeholder] = kaomoji

    return sentence, placeholder_to_kaomoji


def recover_kaomoji(sentences, placeholder_to_kaomoji):
    """
    根据映射表恢复句子中的颜文字。
    Args:
        sentences (list): 含有占位符的句子列表
        placeholder_to_kaomoji (dict): 占位符到颜文字的映射表
    Returns:
        list: 恢复颜文字后的句子列表
    """
    recovered_sentences = []
    for sentence in sentences:
        for placeholder, kaomoji in placeholder_to_kaomoji.items():
            sentence = sentence.replace(placeholder, kaomoji)
        recovered_sentences.append(sentence)
    return recovered_sentences


def get_western_ratio(paragraph):
    """计算段落中字母数字字符的西文比例
    原理：检查段落中字母数字字符的西文比例
    通过is_english_letter函数判断每个字符是否为西文
    只检查字母数字字符，忽略标点符号和空格等非字母数字字符

    Args:
        paragraph: 要检查的文本段落

    Returns:
        float: 西文字符比例(0.0-1.0)，如果没有字母数字字符则返回0.0
    """
    alnum_chars = [char for char in paragraph if char.isalnum()]
    if not alnum_chars:
        return 0.0

    western_count = sum(bool(is_english_letter(char)) for char in alnum_chars)
    return western_count / len(alnum_chars)


def translate_timestamp_to_human_readable(timestamp: float, mode: str = "normal") -> str:
    # sourcery skip: merge-comparisons, merge-duplicate-blocks, switch
    """将时间戳转换为人类可读的时间格式

    Args:
        timestamp: 时间戳
        mode: 转换模式，"normal"为标准格式，"relative"为相对时间格式

    Returns:
        str: 格式化后的时间字符串
    """
    if mode == "normal":
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
    elif mode == "normal_no_YMD":
        return time.strftime("%H:%M:%S", time.localtime(timestamp))
    elif mode == "relative":
        now = time.time()
        diff = now - timestamp

        if diff < 20:
            return "刚刚"
        elif diff < 60:
            return f"{int(diff)}秒前"
        elif diff < 3600:
            return f"{int(diff / 60)}分钟前"
        elif diff < 86400:
            return f"{int(diff / 3600)}小时前"
        elif diff < 86400 * 2:
            return f"{int(diff / 86400)}天前"
        else:
            return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp)) + ":"
    else:  # mode = "lite" or unknown
        # 只返回时分秒格式
        return time.strftime("%H:%M:%S", time.localtime(timestamp))


def get_chat_type_and_target_info(chat_id: str) -> Tuple[bool, Optional["ChatTargetInfo"]]:
    """
    获取聊天类型（是否群聊）和私聊对象信息。

    Args:
        chat_id: 聊天流ID

    Returns:
        Tuple[bool, Optional[Dict]]:
            - bool: 是否为群聊 (True 是群聊, False 是私聊或未知)
            - Optional[Dict]: 如果是私聊，包含对方信息的字典；否则为 None。
            字典包含: platform, user_id, user_nickname, person_id, person_name
    """
    is_group_chat = False  # Default to private/unknown
    chat_target_info = None

    try:
        if chat_stream := _chat_manager.get_session_by_session_id(chat_id):
            if chat_stream.is_group_session:
                is_group_chat = True
                chat_target_info = None  # Explicitly None for group chat
            elif chat_stream.user_id:  # It's a private chat
                is_group_chat = False
                platform: str = chat_stream.platform
                user_id: str = chat_stream.user_id

                # Try to get nickname from context
                user_nickname = None
                if (
                    chat_stream.context
                    and chat_stream.context.message
                    and chat_stream.context.message.message_info.user_info
                ):
                    user_nickname = chat_stream.context.message.message_info.user_info.user_nickname

                from src.common.data_models.chat_target_info_data_model import ChatTargetInfo  # 解决循环导入问题

                # Initialize target_info with basic info
                target_info = ChatTargetInfo(
                    platform=platform,
                    user_id=user_id,
                    session_nickname=user_nickname or "",
                    person_id=None,
                    person_name=None,
                )

                # Try to fetch person info
                try:
                    person = Person(platform=platform, user_id=user_id)
                    if not person.is_known:
                        logger.warning(f"用户 {user_nickname} 尚未认识")
                        # 如果用户尚未认识，则返回False和None
                        return False, None
                    target_info.is_known = True
                    if person.person_id:
                        target_info.person_id = person.person_id
                        target_info.person_name = person.person_name
                except Exception as person_e:
                    logger.warning(
                        f"获取 person_id 或 person_name 时出错 for {platform}:{user_id} in utils: {person_e}"
                    )

                chat_target_info = target_info
        else:
            logger.warning(f"无法获取 chat_stream for {chat_id} in utils")
    except Exception as e:
        logger.error(f"获取聊天类型和目标信息时出错 for {chat_id}: {e}", exc_info=True)

    return is_group_chat, chat_target_info
