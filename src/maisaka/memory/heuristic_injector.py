"""Maisaka 启发式长期记忆与跨窗口自身历史发言自然拉起服务。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from time import time
from typing import Optional, Sequence
import asyncio
import re

from src.chat.message_receive.chat_manager import BotChatSession, chat_manager
from src.chat.message_receive.message import SessionMessage
from src.chat.utils.utils import is_bot_self
from src.common.data_models.message_component_data_model import AtComponent, ReplyComponent
from src.common.logger import get_logger
from src.common.message_repository import count_messages, find_messages
from src.common.prompt_i18n import load_prompt
from src.common.utils.math_utils import format_prompt_short_datetime
from src.config.config import global_config
from src.person_info.person_info import get_person_id
from src.services.llm_service import LLMServiceClient
from src.services.memory_service import MemoryHit, memory_service

logger = get_logger("maisaka_heuristic_memory")

HEURISTIC_MEMORY_REFERENCE_MARKER = "【启发式记忆-内部参考】"
_SOURCE_CHAT_SUMMARY_PREFIX = "chat_summary:"
_SOURCE_PERSON_FACT_PREFIX = "person_fact:"

_SELF_HISTORY_LOOKBACK_SECONDS = 7 * 86400
_SELF_HISTORY_RECENT_FALLBACK_SECONDS = 36 * 3600
_SELF_HISTORY_FETCH_LIMIT = 2500
_SELF_HISTORY_MAX_RELEVANT = 5
_SELF_HISTORY_MAX_RECENT = 2
_SELF_HISTORY_MIN_SCORE = 3.5

_BRACKET_MEDIA_RE = re.compile(r"\[(?:图片|表情包?|动画表情|回复了|引用|视频|语音|文件)[^\]]*(?:\]|$)")
_SIMPLE_BRACKET_RE = re.compile(r"\[[^\[\]]*\]")
_MENTION_RE = re.compile(r"@\S+")
_HISTORICAL_NOISE_RE = re.compile(
    r"不对[，,]直接说|属于是|了这是|真是.{1,8}了$|罢了$|[A-Za-z]{4,}\s+[A-Za-z]{4,}"
)
_NON_WORD_RE = re.compile(r"[^\u4e00-\u9fffA-Za-z0-9]")
_ASCII_TOKEN_RE = re.compile(r"[a-z0-9]{2,}")
_STOP_CHARS = frozenset("的了是在有和就都而及与着或一个我你他她它们这那哪吗呢吧啊呀哦哈啦呗嘛噢喔捏艹草什么啥怎么给让去来点")
_TOPIC_CHARS = frozenset("吃喝玩买抽睡看听穿住行肝氪歪饭菜面粉肉鸡猪牛羊鱼虾茶酒咖啡奶歌剧番游号卡钱班")
_TOPIC_CLUSTERS: tuple[frozenset[str], ...] = (
    frozenset(
        {
            "吃", "喝", "饭", "菜", "面", "粉", "肉", "鸡", "猪", "牛", "羊", "鱼", "虾",
            "饺", "锅", "汤", "粥", "饼", "茶", "酒", "奶", "饿", "饱", "点餐", "食谱",
            "外卖", "食堂", "夜宵", "宵夜", "早饭", "午饭", "晚饭", "螺蛳", "黄焖", "猪脚",
            "麻辣", "沙县", "烤肉", "炸鸡", "肯德基", "麦当劳", "火锅", "烧烤",
        }
    ),
    frozenset(
        {
            "玩", "游", "号", "局", "服", "段位", "匹配", "排位", "坐牢", "三角洲",
            "大红", "护航", "主播", "外挂", "手游", "端游", "开黑", "队友",
        }
    ),
    frozenset({"抽", "歪", "保底", "命", "潜", "专武", "出货", "沉船", "十连", "卡池", "金"}),
    frozenset({"买", "卖", "价", "钱", "降价", "分期", "背刺", "退差价", "耳机", "手机", "显卡"}),
    frozenset({"睡", "困", "早八", "上班", "下班", "加班", "熬夜", "课", "作业"}),
)


def _match_cluster_tokens(text: str, cluster: frozenset[str]) -> set[str]:
    norm = _normalize_colloquial(text)
    return {tok for tok in cluster if tok in norm}


def _clean_history_text(raw_text: str) -> str:
    text = _BRACKET_MEDIA_RE.sub("", str(raw_text or ""))
    prev = None
    while prev != text:
        prev = text
        text = _SIMPLE_BRACKET_RE.sub("", text)
    text = _MENTION_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def _normalize_colloquial(text: str) -> str:
    return (
        str(text or "")
        .lower()
        .replace("啥", "什么")
        .replace("咋", "怎么")
        .replace("哪儿", "哪里")
    )


def _extract_text_features(text: str) -> tuple[set[str], set[str], set[str]]:
    clean = _NON_WORD_RE.sub("", _normalize_colloquial(text))
    chars = {ch for ch in clean if ch not in _STOP_CHARS and "\u4e00" <= ch <= "\u9fff"}
    topic_chars = {ch for ch in clean if ch in _TOPIC_CHARS}
    bigrams = {
        clean[idx : idx + 2]
        for idx in range(len(clean) - 1)
        if not (clean[idx] in _STOP_CHARS and clean[idx + 1] in _STOP_CHARS)
    }
    words = set(_ASCII_TOKEN_RE.findall(clean))
    return chars, topic_chars, bigrams | words


@dataclass
class HeuristicMemoryRecallState:
    """单个聊天流的启发式记忆运行时状态。"""

    last_recall_at: float = 0.0
    last_message_count: int = 0
    cached_reference: str = ""
    cache_expires_at: float = 0.0
    last_focus_key: str = ""


@dataclass(frozen=True)
class HeuristicMemoryContext:
    """一次启发式记忆召回的上下文。"""

    session: BotChatSession
    recent_messages: list[SessionMessage]
    active_person_ids: set[str] = field(default_factory=set)
    total_message_count: int = 0
    focus_messages: list[SessionMessage] = field(default_factory=list)
    focus_query: str = ""
    window_start_time: float = 0.0


@dataclass
class _SelfReplyExchange:
    time_label: str
    timestamp: float
    prompt_uid: str
    prompt_nick: str
    prompt_text: str
    prompt_directed: bool
    bot_text: str
    followup_text: str = ""


def _format_exchange_line(index: int, ex: _SelfReplyExchange) -> str:
    prefix = f"{index}. [{ex.time_label}] "
    prompt_part = f"{ex.prompt_nick}：“{ex.prompt_text}” → " if ex.prompt_text else ""
    followup_part = f"（{ex.followup_text}）" if ex.followup_text else ""
    return f"{prefix}{prompt_part}你曾回：“{ex.bot_text}”{followup_part}"


class HeuristicMemoryInjector:
    """根据当前聊天流印象与焦点消息自然拉起长期记忆与跨窗口自身历史发言。"""

    def __init__(self) -> None:
        self._states: dict[str, HeuristicMemoryRecallState] = {}
        self._impression_client = LLMServiceClient(
            task_name="utils",
            request_type="heuristic_memory_impression",
        )

    async def build_injection_message(
        self,
        *,
        session_id: str,
        anchor_message: Optional[SessionMessage] = None,
        source_messages: Optional[Sequence[SessionMessage]] = None,
    ) -> str:
        """构造给 Planner/Replyer 共享的一次性启发式记忆与跨窗口自身发言参考。"""

        config = global_config.a_memorix.integration
        if not config.heuristic_memory_recall_enabled:
            self.clear_session_reference(session_id)
            return ""

        session = chat_manager.get_existing_session_by_session_id(session_id)
        if session is None:
            logger.debug(f"启发式记忆跳过：无法解析真实聊天流 session_id={session_id!r}")
            return ""

        window_size = max(1, config.heuristic_memory_recall_window_size)
        total_message_count = count_messages(session_id=session.session_id)
        if total_message_count <= 0:
            self.clear_session_reference(session.session_id)
            return ""

        recent_messages = find_messages(
            session_id=session.session_id,
            limit=window_size,
            limit_mode="latest",
            filter_command=True,
        )
        if not recent_messages:
            self.clear_session_reference(session.session_id)
            return ""

        focus_msgs = self._collect_focus_messages(anchor_message, source_messages, recent_messages)
        focus_query = self._build_focus_query(focus_msgs)
        window_start_time = self._resolve_window_start_time(recent_messages)

        context = HeuristicMemoryContext(
            session=session,
            recent_messages=recent_messages,
            active_person_ids=self._collect_active_person_ids(recent_messages),
            total_message_count=total_message_count,
            focus_messages=focus_msgs,
            focus_query=focus_query,
            window_start_time=window_start_time,
        )

        long_term_section = await self._resolve_long_term_section(context)
        self_history_section = self._build_self_reply_history_section(context)

        sections = [sec for sec in (long_term_section, self_history_section) if sec.strip()]
        if not sections:
            return ""
        return f"{HEURISTIC_MEMORY_REFERENCE_MARKER}\n" + "\n\n".join(sections)

    async def _resolve_long_term_section(self, context: HeuristicMemoryContext) -> str:
        config = global_config.a_memorix.integration
        session_id = context.session.session_id
        state = self._states.setdefault(session_id, HeuristicMemoryRecallState())
        now = time()
        cache_ttl = config.heuristic_memory_recall_cache_ttl_seconds
        focus_key = context.focus_query.strip()
        same_focus = not focus_key or focus_key == state.last_focus_key

        if state.cached_reference and cache_ttl > 0 and now < state.cache_expires_at and same_focus:
            return state.cached_reference

        can_trigger = self._can_trigger(
            state=state,
            total_message_count=context.total_message_count,
            now=now,
            min_interval_seconds=config.heuristic_memory_recall_min_interval_seconds,
            min_new_messages=config.heuristic_memory_recall_min_new_messages,
            focus_changed=bool(focus_key and focus_key != state.last_focus_key),
        )
        if not can_trigger:
            if focus_key and focus_key != state.last_focus_key:
                try:
                    hits = await self._search_related_memory(focus_key, context)
                    direct_section = self._format_long_term_section(
                        hits,
                        max_chars=config.heuristic_memory_recall_max_chars,
                    )
                    if direct_section:
                        state.last_focus_key = focus_key
                        state.cached_reference = direct_section
                        state.cache_expires_at = now + cache_ttl if cache_ttl > 0 else 0.0
                        return direct_section
                except Exception as exc:
                    logger.debug(f"启发式焦点记忆直搜失败，已跳过: {exc}")
            if state.cached_reference and cache_ttl > 0 and now < state.cache_expires_at:
                return state.cached_reference
            return ""

        try:
            impression = await self._build_chat_impression(context)
            query = impression or focus_key
            if not query:
                self.clear_session_reference(session_id)
                return ""
            hits = await self._search_related_memory(query, context)
            section = self._format_long_term_section(
                hits,
                max_chars=config.heuristic_memory_recall_max_chars,
            )
        except Exception as exc:
            self.clear_session_reference(session_id)
            logger.debug(f"启发式记忆自然拉起失败，已跳过: {exc}", exc_info=True)
            return ""

        state.last_recall_at = now
        state.last_message_count = context.total_message_count
        state.last_focus_key = focus_key
        state.cached_reference = section
        state.cache_expires_at = now + cache_ttl if (section and cache_ttl > 0) else 0.0
        if section:
            logger.info(
                f"启发式记忆自然拉起成功: session_id={session_id} "
                f"hits={len(hits)} total_messages={context.total_message_count}"
            )
        return section

    def clear_session_reference(self, session_id: str) -> None:
        """清理当前会话的本轮注入参考。"""

        state = self._states.get(str(session_id or "").strip())
        if state is None:
            return
        state.cached_reference = ""
        state.cache_expires_at = 0.0

    @staticmethod
    def _can_trigger(
        *,
        state: HeuristicMemoryRecallState,
        total_message_count: int,
        now: float,
        min_interval_seconds: int,
        min_new_messages: int,
        focus_changed: bool = False,
    ) -> bool:
        if state.last_message_count <= 0:
            return True
        elapsed = now - state.last_recall_at
        if focus_changed and elapsed >= min(30, max(0, min_interval_seconds)):
            return True
        if min_interval_seconds > 0 and elapsed < min_interval_seconds:
            return False
        return total_message_count - state.last_message_count >= min_new_messages

    @staticmethod
    def _collect_focus_messages(
        anchor_message: Optional[SessionMessage],
        source_messages: Optional[Sequence[SessionMessage]],
        recent_messages: Sequence[SessionMessage],
    ) -> list[SessionMessage]:
        items: list[SessionMessage] = []
        seen_ids: set[str] = set()
        for msg in [*(source_messages or []), anchor_message]:
            if msg is None:
                continue
            mid = str(getattr(msg, "message_id", "") or "")
            if mid and mid in seen_ids:
                continue
            if mid:
                seen_ids.add(mid)
            items.append(msg)
        if items:
            return items[-3:]
        human_recent = [
            msg
            for msg in recent_messages
            if not is_bot_self(str(getattr(msg, "platform", "") or ""), str(getattr(msg.message_info.user_info, "user_id", "") or ""))
        ]
        return human_recent[-2:]

    @classmethod
    def _build_focus_query(cls, focus_messages: Sequence[SessionMessage]) -> str:
        parts: list[str] = []
        for msg in focus_messages[-3:]:
            cleaned = _clean_history_text(getattr(msg, "processed_plain_text", ""))
            if cleaned and cleaned not in parts:
                parts.append(cleaned)
        return " ".join(parts).strip()

    @classmethod
    def _resolve_window_start_time(cls, recent_messages: Sequence[SessionMessage]) -> float:
        earliest = 0.0
        for msg in recent_messages:
            ts = cls._extract_timestamp(msg)
            if ts > 0 and (earliest <= 0 or ts < earliest):
                earliest = ts
        return earliest

    @staticmethod
    def _extract_timestamp(message: SessionMessage) -> float:
        raw_time = getattr(message, "time", None) or getattr(message, "timestamp", None)
        if isinstance(raw_time, (int, float)):
            return float(raw_time)
        if isinstance(raw_time, datetime):
            return raw_time.timestamp()
        return 0.0

    async def _build_chat_impression(self, context: HeuristicMemoryContext) -> str:
        prompt = load_prompt(
            "heuristic_memory_impression",
            chat_identity=self._format_chat_identity(context.session),
            message_window=self._format_message_window(context.recent_messages, context.focus_messages),
        )
        result = await self._impression_client.generate_response(prompt)
        return str(result.response or "").strip()

    async def _search_related_memory(
        self,
        impression: str,
        context: HeuristicMemoryContext,
    ) -> list[MemoryHit]:
        config = global_config.a_memorix.integration
        limit = config.heuristic_memory_recall_limit
        search_limit = min(20, limit * 4)
        cross_chat_enabled = config.heuristic_memory_cross_chat_enabled
        chat_id = "" if cross_chat_enabled else context.session.session_id
        user_id = str(context.session.user_id or "")
        group_id = str(context.session.group_id or "")

        focus_query = context.focus_query.strip()
        if focus_query and focus_query != impression.strip():
            primary_res, focus_res = await asyncio.gather(
                memory_service.search(
                    impression,
                    limit=search_limit,
                    mode="search",
                    chat_id=chat_id,
                    person_id="",
                    respect_filter=not cross_chat_enabled,
                    user_id=user_id,
                    group_id=group_id,
                ),
                memory_service.search(
                    focus_query,
                    limit=search_limit,
                    mode="search",
                    chat_id=chat_id,
                    person_id="",
                    respect_filter=not cross_chat_enabled,
                    user_id=user_id,
                    group_id=group_id,
                ),
            )
            raw_hits = self._merge_raw_hits(
                primary_res.hits if (primary_res.success and not primary_res.filtered) else [],
                focus_res.hits if (focus_res.success and not focus_res.filtered) else [],
            )
        else:
            result = await memory_service.search(
                impression,
                limit=search_limit,
                mode="search",
                chat_id=chat_id,
                person_id="",
                respect_filter=not cross_chat_enabled,
                user_id=user_id,
                group_id=group_id,
            )
            raw_hits = result.hits if (result.success and not result.filtered) else []

        if not raw_hits:
            return []

        resolved_sources = await self._resolve_hit_sources_by_hash(raw_hits)
        filtered: list[MemoryHit] = []
        seen_hashes: set[str] = set()
        for hit in raw_hits:
            resolved_source = resolved_sources.get(hit.hash_value, "")
            if not self._is_hit_allowed(hit, context, resolved_source=resolved_source):
                continue
            dedup_key = hit.hash_value or hit.content
            if dedup_key in seen_hashes:
                continue
            seen_hashes.add(dedup_key)
            filtered.append(hit)
            if len(filtered) >= limit:
                break
        return filtered

    @staticmethod
    def _merge_raw_hits(
        primary_hits: Sequence[MemoryHit],
        focus_hits: Sequence[MemoryHit],
    ) -> list[MemoryHit]:
        by_key: dict[str, MemoryHit] = {}
        for item in primary_hits:
            key = item.hash_value or item.content.strip()
            if key:
                by_key[key] = item
        for item in focus_hits:
            key = item.hash_value or item.content.strip()
            if not key:
                continue
            boosted = MemoryHit(
                content=item.content,
                score=float(item.score or 0.0) + 0.08,
                hit_type=item.hit_type,
                source=item.source,
                hash_value=item.hash_value,
                metadata=item.metadata,
                episode_id=item.episode_id,
                title=item.title,
            )
            existing = by_key.get(key)
            if existing is None or boosted.score > existing.score:
                by_key[key] = boosted
        return sorted(by_key.values(), key=lambda x: float(x.score or 0.0), reverse=True)

    async def _resolve_hit_sources_by_hash(self, hits: Sequence[MemoryHit]) -> dict[str, str]:
        paragraph_hashes = [hit.hash_value for hit in hits if hit.hash_value and not self._hit_has_scope_hint(hit)]
        if not paragraph_hashes:
            return {}

        try:
            payload = await memory_service.delete_admin(
                action="preview",
                mode="paragraph",
                selector={"hashes": paragraph_hashes},
                timeout_ms=10000,
            )
        except Exception as exc:
            logger.debug(f"启发式记忆命中来源解析失败，已按未知来源处理: {exc}")
            return {}
        if not isinstance(payload, dict) or not bool(payload.get("success", False)):
            return {}

        sources: dict[str, str] = {}
        for item in payload.get("items", []) or []:
            if not isinstance(item, dict):
                continue
            if str(item.get("item_type", "") or "").strip() != "paragraph":
                continue
            paragraph_hash = str(item.get("item_hash", "") or "").strip()
            source = str(item.get("source", "") or "").strip()
            if paragraph_hash and source:
                sources[paragraph_hash] = source
        return sources

    @staticmethod
    def _hit_has_scope_hint(hit: MemoryHit) -> bool:
        metadata = hit.metadata if isinstance(hit.metadata, dict) else {}
        if any(metadata.get(key) for key in ("source", "chat_id", "person_id", "person_ids")):
            return True
        source = str(hit.source or "").strip()
        return source.startswith(_SOURCE_CHAT_SUMMARY_PREFIX) or source.startswith(_SOURCE_PERSON_FACT_PREFIX)

    def _is_hit_allowed(
        self,
        hit: MemoryHit,
        context: HeuristicMemoryContext,
        *,
        resolved_source: str = "",
    ) -> bool:
        metadata = hit.metadata if isinstance(hit.metadata, dict) else {}
        source = str(metadata.get("source") or resolved_source or hit.source or "").strip()
        source_type = str(metadata.get("source_type") or "").strip()
        chat_id = str(metadata.get("chat_id") or "").strip()
        person_id = str(metadata.get("person_id") or "").strip()
        if not person_id:
            person_ids = metadata.get("person_ids")
            if isinstance(person_ids, list):
                person_id = next((str(item).strip() for item in person_ids if str(item).strip()), "")
        if not person_id and source.startswith(_SOURCE_PERSON_FACT_PREFIX):
            person_id = source[len(_SOURCE_PERSON_FACT_PREFIX) :].strip()

        if hit.hit_type == "episode":
            episode_source = str(metadata.get("source") or "").strip()
            return episode_source == f"{_SOURCE_CHAT_SUMMARY_PREFIX}{context.session.session_id}"

        if source_type == "person_fact" or source.startswith(_SOURCE_PERSON_FACT_PREFIX):
            return bool(person_id and person_id in context.active_person_ids)

        if source_type == "chat_summary" or source.startswith(_SOURCE_CHAT_SUMMARY_PREFIX):
            source_session_id = self._resolve_source_session_id(source=source, chat_id=chat_id)
            return self._is_chat_memory_allowed(source_session_id, context.session)

        if chat_id:
            return self._is_chat_memory_allowed(chat_id, context.session)
        return False

    @staticmethod
    def _resolve_source_session_id(*, source: str, chat_id: str) -> str:
        if chat_id:
            return chat_id
        if source.startswith(_SOURCE_CHAT_SUMMARY_PREFIX):
            return source[len(_SOURCE_CHAT_SUMMARY_PREFIX) :].strip()
        return ""

    def _is_chat_memory_allowed(self, source_session_id: str, current_session: BotChatSession) -> bool:
        clean_source_session_id = str(source_session_id or "").strip()
        if not clean_source_session_id:
            return False
        if clean_source_session_id == current_session.session_id:
            return True

        config = global_config.a_memorix.integration
        if not config.heuristic_memory_cross_chat_enabled:
            return False

        source_session = chat_manager.get_existing_session_by_session_id(clean_source_session_id)
        if source_session is None:
            return False

        if source_session.is_group_session and not current_session.is_group_session:
            return config.heuristic_memory_group_to_private_enabled
        if not source_session.is_group_session and current_session.is_group_session:
            return config.heuristic_memory_private_to_group_enabled
        return True

    @staticmethod
    def _collect_active_person_ids(messages: Sequence[SessionMessage]) -> set[str]:
        person_ids: set[str] = set()
        for message in messages:
            platform = str(getattr(message, "platform", "") or "").strip()
            user_id = str(getattr(message.message_info.user_info, "user_id", "") or "").strip()
            if platform and user_id:
                person_ids.add(get_person_id(platform, user_id))

            raw_message = getattr(message, "raw_message", None)
            for component in getattr(raw_message, "components", []) or []:
                if isinstance(component, AtComponent):
                    target_user_id = str(component.target_user_id or "").strip()
                    if platform and target_user_id:
                        person_ids.add(get_person_id(platform, target_user_id))
                elif isinstance(component, ReplyComponent):
                    target_user_id = str(component.target_message_sender_id or "").strip()
                    if platform and target_user_id:
                        person_ids.add(get_person_id(platform, target_user_id))
        return person_ids

    @staticmethod
    def _format_chat_identity(session: BotChatSession) -> str:
        chat_type = "group" if session.is_group_session else "private"
        display_name = chat_manager.get_session_name(session.session_id) or session.session_id
        parts = [
            f"chat_type: {chat_type}",
            f"display_name: {display_name}",
            f"platform: {session.platform}",
        ]
        if session.group_id:
            parts.append(f"group_id: {session.group_id}")
        if session.user_id:
            parts.append(f"user_id: {session.user_id}")
        return "\n".join(parts)

    @classmethod
    def _format_message_window(
        cls,
        messages: Sequence[SessionMessage],
        focus_messages: Sequence[SessionMessage] = (),
    ) -> str:
        lines: list[str] = []
        for message in messages:
            sender = cls._message_sender_name(message)
            text = str(getattr(message, "processed_plain_text", "") or "").strip()
            if not text:
                continue
            text = text.replace("\r", " ").replace("\n", " ")
            if len(text) > 220:
                text = text[:220].rstrip() + "..."
            lines.append(f"- {sender}: {text}")
        window_str = "\n".join(lines) if lines else "no text messages available"
        focus_lines: list[str] = []
        for msg in focus_messages[-3:]:
            sender = cls._message_sender_name(msg)
            cleaned = _clean_history_text(getattr(msg, "processed_plain_text", ""))
            if cleaned:
                focus_lines.append(f"- {sender}: {cleaned}")
        if focus_lines:
            window_str += "\n\n【当前触发焦点（以此为核心生成检索印象）】\n" + "\n".join(focus_lines)
        return window_str

    @staticmethod
    def _message_sender_name(message: SessionMessage) -> str:
        user_info = message.message_info.user_info
        return str(
            getattr(user_info, "user_cardname", "")
            or getattr(user_info, "user_nickname", "")
            or getattr(user_info, "user_id", "")
            or "unknown_user"
        ).strip()

    @staticmethod
    def _format_long_term_section(hits: Sequence[MemoryHit], *, max_chars: int) -> str:
        if not hits:
            return ""
        lines = [
            "【关联长期记忆-内部参考（仅作背景记忆，请勿原样复述）】",
        ]
        for index, hit in enumerate(hits, start=1):
            content = str(hit.content or "").strip().replace("\r", " ").replace("\n", " ")
            if not content:
                continue
            if len(content) > 180:
                content = content[:180].rstrip() + "..."
            lines.append(f"{index}. {content}")

        if len(lines) <= 1:
            return ""
        section = "\n".join(lines).strip()
        if len(section) <= max_chars:
            return section
        return section[:max_chars].rstrip() + "..."

    @classmethod
    def _format_reference(cls, hits: Sequence[MemoryHit], *, max_chars: int) -> str:
        section = cls._format_long_term_section(hits, max_chars=max_chars)
        if not section:
            return ""
        return f"{HEURISTIC_MEMORY_REFERENCE_MARKER}\n{section}"

    def _build_self_reply_history_section(self, context: HeuristicMemoryContext) -> str:
        now = time()
        history_messages = find_messages(
            session_id=context.session.session_id,
            start_time=max(0.0, now - _SELF_HISTORY_LOOKBACK_SECONDS),
            before_time=context.window_start_time if context.window_start_time > 0 else None,
            limit=_SELF_HISTORY_FETCH_LIMIT,
            limit_mode="latest",
            filter_command=True,
        )
        if not history_messages:
            return ""

        exchanges = self._extract_self_reply_exchanges(history_messages)
        if not exchanges:
            return ""

        selected = self._select_self_reply_exchanges(
            exchanges=exchanges,
            focus_query=context.focus_query,
            now=now,
        )
        if not selected:
            return ""

        lines = [_format_exchange_line(idx, item) for idx, item in enumerate(selected, start=1)]
        return (
            "【你近期在本群的历史发言回顾-内部参考（跨天/前序窗口已说过的话，"
            "遇到同类提问、推荐或接梗时必须换新内容，严禁重复报同样的菜名、推荐项或车轱辘话）】\n"
            + "\n".join(lines)
        )

    def _extract_self_reply_exchanges(
        self,
        messages: Sequence[SessionMessage],
    ) -> list[_SelfReplyExchange]:
        by_id = {str(msg.message_id): msg for msg in messages if getattr(msg, "message_id", None)}
        exchanges: list[_SelfReplyExchange] = []
        consumed_indices: set[int] = set()
        idx = 0
        total = len(messages)

        while idx < total:
            if idx in consumed_indices:
                idx += 1
                continue
            msg = messages[idx]
            if not is_bot_self(str(getattr(msg, "platform", "") or ""), str(getattr(msg.message_info.user_info, "user_id", "") or "")):
                idx += 1
                continue

            bot_parts, bot_mids, first_rto, last_ts, next_idx = self._collect_bot_turn(
                messages, idx, consumed_indices
            )
            if not bot_parts:
                idx = next_idx
                continue

            ts = self._extract_timestamp(msg)
            p_uid, p_nick, p_text, p_directed = self._find_prompt_for_bot_turn(
                messages, idx, ts, first_rto, by_id
            )
            followup_text = self._find_followup_for_bot_turn(
                messages, next_idx, last_ts, bot_parts, bot_mids, consumed_indices
            )
            time_label = format_prompt_short_datetime(ts)
            exchanges.append(
                _SelfReplyExchange(
                    time_label=time_label,
                    timestamp=ts,
                    prompt_uid=p_uid,
                    prompt_nick=p_nick,
                    prompt_text=p_text,
                    prompt_directed=p_directed,
                    bot_text=" / ".join(bot_parts),
                    followup_text=followup_text,
                )
            )
            idx = next_idx

        return exchanges

    def _collect_bot_turn(
        self,
        messages: Sequence[SessionMessage],
        start_idx: int,
        consumed_indices: set[int],
    ) -> tuple[list[str], set[str], str, float, int]:
        first_msg = messages[start_idx]
        first_ts = self._extract_timestamp(first_msg)
        first_rto = str(getattr(first_msg, "reply_to", "") or "").strip()
        bot_parts: list[str] = []
        bot_mids: set[str] = set()
        last_ts = first_ts
        cur = start_idx

        while cur < len(messages):
            if cur in consumed_indices:
                cur += 1
                continue
            cand = messages[cur]
            if not is_bot_self(str(getattr(cand, "platform", "") or ""), str(getattr(cand.message_info.user_info, "user_id", "") or "")):
                break
            cand_ts = self._extract_timestamp(cand)
            cand_rto = str(getattr(cand, "reply_to", "") or "").strip()
            if cur > start_idx and (
                (cand_ts - last_ts) > 25.0
                or (cand_rto and first_rto and cand_rto != first_rto)
            ):
                break
            if getattr(cand, "message_id", None):
                bot_mids.add(str(cand.message_id))
            if not getattr(cand, "is_emoji", False) and not getattr(cand, "is_picture", False):
                cleaned = _clean_history_text(getattr(cand, "processed_plain_text", ""))
                if cleaned and len(cleaned) <= 80 and not _HISTORICAL_NOISE_RE.search(cleaned):
                    if cleaned not in bot_parts:
                        bot_parts.append(cleaned)
            if not first_rto and cand_rto:
                first_rto = cand_rto
            last_ts = cand_ts
            cur += 1

        return bot_parts, bot_mids, first_rto, last_ts, cur

    def _find_prompt_for_bot_turn(
        self,
        messages: Sequence[SessionMessage],
        bot_idx: int,
        bot_ts: float,
        reply_to_id: str,
        by_id: dict[str, SessionMessage],
    ) -> tuple[str, str, str, bool]:
        target_msg = by_id.get(reply_to_id) if reply_to_id else None
        if target_msg is None:
            for back_idx in range(bot_idx - 1, max(-1, bot_idx - 6), -1):
                cand = messages[back_idx]
                if is_bot_self(str(getattr(cand, "platform", "") or ""), str(getattr(cand.message_info.user_info, "user_id", "") or "")):
                    continue
                if (bot_ts - self._extract_timestamp(cand)) > 180.0:
                    break
                if _clean_history_text(getattr(cand, "processed_plain_text", "")):
                    target_msg = cand
                    break
        if target_msg is None:
            return "", "", "", False

        cleaned = _clean_history_text(getattr(target_msg, "processed_plain_text", ""))[:36]
        if not cleaned:
            return "", "", "", False
        uinfo = target_msg.message_info.user_info
        uid = str(getattr(uinfo, "user_id", "") or "")
        nick = str(getattr(uinfo, "user_nickname", "") or uid or "群友").strip()
        raw_text = str(getattr(target_msg, "processed_plain_text", "") or "")
        directed = bool(
            getattr(target_msg, "is_at", False)
            or getattr(target_msg, "is_mentioned", False)
            or "🍗" in raw_text
            or "麦" in raw_text
        )
        return uid, nick, cleaned, directed

    def _find_followup_for_bot_turn(
        self,
        messages: Sequence[SessionMessage],
        after_idx: int,
        last_bot_ts: float,
        bot_parts: Sequence[str],
        bot_mids: set[str],
        consumed_indices: set[int],
    ) -> str:
        followup_nick = ""
        followup_parts: list[str] = []
        followup_mids: set[str] = set()
        followup_uid = ""

        for fwd_idx in range(after_idx, min(len(messages), after_idx + 18)):
            cand = messages[fwd_idx]
            cand_ts = self._extract_timestamp(cand)
            if (cand_ts - last_bot_ts) > 180.0:
                break
            cand_uid = str(getattr(cand.message_info.user_info, "user_id", "") or "")
            if is_bot_self(str(getattr(cand, "platform", "") or ""), cand_uid):
                cand_rto_bot = str(getattr(cand, "reply_to", "") or "").strip()
                if followup_parts and (not cand_rto_bot or cand_rto_bot in followup_mids) and (cand_ts - last_bot_ts) <= 90.0:
                    next_parts, _, _, _, next_end = self._collect_bot_turn(messages, fwd_idx, consumed_indices)
                    for c_i in range(fwd_idx, next_end):
                        consumed_indices.add(c_i)
                    if next_parts:
                        joined_fb = " / ".join(followup_parts)
                        joined_next = " / ".join(next_parts)
                        return f"群友反馈：{followup_nick}：“{joined_fb}” → 你又回：“{joined_next}”"
                continue

            raw_ft = str(getattr(cand, "processed_plain_text", "") or "")
            cleaned_ft = _clean_history_text(raw_ft)[:28]
            if not cleaned_ft:
                continue
            cand_rto = str(getattr(cand, "reply_to", "") or "").strip()
            quoted_bot = cand_rto in bot_mids or any(bp[:5] and bp[:5] in raw_ft for bp in bot_parts)
            same_user_followup = bool(
                followup_uid
                and cand_uid == followup_uid
                and not cand_rto
                and "@" not in raw_ft
                and len(followup_parts) < 2
            )
            if quoted_bot or same_user_followup:
                if not followup_nick:
                    followup_nick = str(getattr(cand.message_info.user_info, "user_nickname", "") or cand_uid or "群友").strip()
                    followup_uid = cand_uid
                if cleaned_ft not in followup_parts:
                    followup_parts.append(cleaned_ft)
                if getattr(cand, "message_id", None):
                    followup_mids.add(str(cand.message_id))

        if not followup_parts:
            return ""
        return f"群友反馈：{followup_nick}：“{' / '.join(followup_parts)}”"

    @staticmethod
    def _select_self_reply_exchanges(
        *,
        exchanges: Sequence[_SelfReplyExchange],
        focus_query: str,
        now: float,
    ) -> list[_SelfReplyExchange]:
        selected: list[_SelfReplyExchange] = []
        seen_bot_texts: set[str] = set()

        if focus_query.strip():
            f_chars, f_topic, f_bg = _extract_text_features(focus_query)
            active_clusters = [c for c in _TOPIC_CLUSTERS if _match_cluster_tokens(focus_query, c)]
            scored: list[tuple[float, float, _SelfReplyExchange]] = []
            for ex in exchanges:
                cand_text = f"{ex.prompt_text} {ex.bot_text} {ex.followup_text}"
                c_chars, c_topic, c_bg = _extract_text_features(cand_text)
                score = (
                    len(f_bg & c_bg) * 3.0
                    + len(f_topic & c_topic) * 3.5
                    + len(f_chars & c_chars) * 1.0
                )
                for cluster in active_clusters:
                    bot_cluster_hits = _match_cluster_tokens(f"{ex.bot_text} {ex.followup_text}", cluster)
                    if bot_cluster_hits:
                        score += min(10.0, len(bot_cluster_hits) * 2.5)
                if score > 0 and ex.prompt_directed:
                    score += 1.5
                if score > 0 and ex.followup_text:
                    score += 2.5
                if score >= _SELF_HISTORY_MIN_SCORE:
                    scored.append((score, ex.timestamp, ex))
            scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
            for _, _, ex in scored:
                norm_bot = ex.bot_text.strip()
                if norm_bot in seen_bot_texts:
                    continue
                seen_bot_texts.add(norm_bot)
                selected.append(ex)
                if len(selected) >= _SELF_HISTORY_MAX_RELEVANT:
                    break

        recent_added = 0
        for ex in reversed(exchanges):
            if recent_added >= _SELF_HISTORY_MAX_RECENT or len(selected) >= (_SELF_HISTORY_MAX_RELEVANT + 1):
                break
            if ex.timestamp > 0 and (now - ex.timestamp) > _SELF_HISTORY_RECENT_FALLBACK_SECONDS:
                continue
            norm_bot = ex.bot_text.strip()
            if norm_bot in seen_bot_texts:
                continue
            seen_bot_texts.add(norm_bot)
            selected.append(ex)
            recent_added += 1

        return selected


heuristic_memory_injector = HeuristicMemoryInjector()
