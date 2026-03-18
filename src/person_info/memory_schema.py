import hashlib
import math
import re
import time
from typing import Any, Callable, Iterable, List, Optional, TypedDict, Union

from src.common.database.database_model import PersonInfo
from src.common.logger import get_logger

logger = get_logger("person_info")

DEFAULT_MEMORY_CATEGORY = "其他"
MAX_MEMORY_POINTS = 200
MAX_MEMORY_POINTS_PER_CATEGORY = 30
MEMORY_SIMILARITY_THRESHOLD = 0.9
MEMORY_MIN_WEIGHT = 0.05
MEMORY_MAX_WEIGHT = 5.0
MEMORY_DECAY_GRACE_SECONDS = 6 * 60 * 60
MEMORY_DECAY_TIMESPAN = 14 * 24 * 60 * 60
MEMORY_RECENCY_TIMESPAN = 7 * 24 * 60 * 60
MEMORY_FORGET_THRESHOLD = 0.15
MEMORY_HARD_FORGET_SECONDS = 180 * 24 * 60 * 60

CATEGORY_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("喜好", ("喜欢", "爱吃", "偏好", "最爱", "爱喝", "爱看")),
    ("经历", ("经历", "发生", "去了", "参加", "上次", "以前")),
    ("性格", ("性格", "脾气", "内向", "外向", "自信", "害羞")),
    ("能力", ("擅长", "会", "技能", "专业", "懂", "熟悉")),
]


class MemoryPointDict(TypedDict, total=False):
    id: str
    category: str
    content: str
    weight: float
    created_at: float
    updated_at: float
    last_used_at: float
    hits: int
    source: Optional[str]
    context: Optional[str]


def _current_timestamp() -> float:
    return time.time()


def _normalize_category(category: str) -> str:
    category = (category or "").strip()
    return category if category else DEFAULT_MEMORY_CATEGORY


def _normalize_weight(weight: float | None) -> float:
    try:
        value = float(weight)
    except (TypeError, ValueError):
        value = 1.0
    return round(min(max(value, MEMORY_MIN_WEIGHT), MEMORY_MAX_WEIGHT), 2)


def _legacy_memory_to_dict(memory_point: str) -> Optional[MemoryPointDict]:
    if not isinstance(memory_point, str):
        return None
    parts = memory_point.split(":")
    if len(parts) < 3:
        return None

    category = _normalize_category(parts[0])
    content = ":".join(parts[1:-1]).strip()
    if not content:
        return None

    now = _current_timestamp()
    return MemoryPointDict(
        category=category,
        content=content,
        weight=_normalize_weight(parts[-1]),
        created_at=now,
        updated_at=now,
        last_used_at=now,
        hits=1,
    )


def _normalize_memory_dict(memory_point: Any) -> Optional[MemoryPointDict]:
    if isinstance(memory_point, dict):
        now = _current_timestamp()
        category = _normalize_category(memory_point.get("category", ""))
        content = (memory_point.get("content") or "").strip()
        if not content:
            return None

        normalized = MemoryPointDict(
            category=category,
            content=content,
            weight=_normalize_weight(memory_point.get("weight")),
            created_at=float(memory_point.get("created_at") or now),
            updated_at=float(memory_point.get("updated_at") or now),
            last_used_at=float(memory_point.get("last_used_at") or memory_point.get("created_at") or now),
            hits=int(memory_point.get("hits") or 1),
        )
        if memory_point.get("id"):
            normalized["id"] = str(memory_point.get("id"))
        if memory_point.get("source"):
            normalized["source"] = str(memory_point.get("source"))
        if memory_point.get("context"):
            normalized["context"] = str(memory_point.get("context"))
        return normalized

    if isinstance(memory_point, str):
        return _legacy_memory_to_dict(memory_point)
    return None


def _apply_memory_decay(memory_point: MemoryPointDict, now: Optional[float] = None) -> None:
    if not memory_point:
        return

    now = now or _current_timestamp()
    last_used = (
        memory_point.get("last_used_at") or memory_point.get("updated_at") or memory_point.get("created_at") or now
    )
    elapsed = max(0.0, now - last_used)
    if elapsed <= MEMORY_DECAY_GRACE_SECONDS:
        return

    decay_factor = math.exp(-elapsed / MEMORY_DECAY_TIMESPAN)
    memory_point["weight"] = max(MEMORY_MIN_WEIGHT, round(memory_point.get("weight", 1.0) * decay_factor, 4))
    memory_point["updated_at"] = now


def _should_forget_memory(memory_point: MemoryPointDict, now: Optional[float] = None) -> bool:
    now = now or _current_timestamp()
    last_used = (
        memory_point.get("last_used_at") or memory_point.get("updated_at") or memory_point.get("created_at") or now
    )
    age_since_use = now - last_used
    age_since_create = now - (memory_point.get("created_at") or now)
    if memory_point.get("weight", 0.0) <= MEMORY_FORGET_THRESHOLD and age_since_use > MEMORY_DECAY_GRACE_SECONDS:
        return True
    if age_since_create > MEMORY_HARD_FORGET_SECONDS:
        return True
    return False


def _sanitize_memory_points(memory_points: Iterable[Any]) -> List[MemoryPointDict]:
    now = _current_timestamp()
    sanitized: List[MemoryPointDict] = []
    for point in memory_points:
        normalized = _normalize_memory_dict(point)
        if not normalized:
            continue
        _apply_memory_decay(normalized, now=now)
        if _should_forget_memory(normalized, now=now):
            continue
        sanitized.append(normalized)
    return sanitized


def _memory_to_log_string(memory_point: MemoryPointDict) -> str:
    category = memory_point.get("category") or DEFAULT_MEMORY_CATEGORY
    content = memory_point.get("content") or ""
    weight = memory_point.get("weight", 0.0)
    return f"{category}:{content}:{weight:.2f}"


def _remove_lowest_weight_entry(
    memory_points: List[MemoryPointDict],
    *,
    predicate: Optional[Callable[[MemoryPointDict], bool]] = None,
) -> Optional[MemoryPointDict]:
    predicate = predicate or (lambda _p: True)
    lowest_idx = None
    lowest_weight = math.inf
    for idx, point in enumerate(memory_points):
        if not predicate(point):
            continue
        weight = point.get("weight", math.inf)
        if weight < lowest_weight:
            lowest_weight = weight
            lowest_idx = idx

    if lowest_idx is None:
        return None
    return memory_points.pop(lowest_idx)


def infer_memory_category(memory_content: str) -> str:
    memory_content = (memory_content or "").strip()
    lowered = memory_content.lower()
    for category, keywords in CATEGORY_KEYWORDS:
        if any(keyword.lower() in lowered for keyword in keywords):
            return category
    return DEFAULT_MEMORY_CATEGORY


def get_person_id(platform: str, user_id: Union[int, str]) -> str:
    if "-" in platform:
        platform = platform.split("-")[1]
    key = "_".join([platform, str(user_id)])
    return hashlib.md5(key.encode()).hexdigest()


def get_person_id_by_person_name(person_name: str) -> str:
    try:
        record = PersonInfo.get_or_none(PersonInfo.person_name == person_name)
        return record.person_id if record else ""
    except Exception as exc:
        logger.error(f"根据用户名 {person_name} 获取用户ID时出错 (Peewee): {exc}")
        return ""


def get_person_id_by_alias(alias: str) -> str:
    alias = (alias or "").strip()
    if not alias:
        return ""

    try:
        record = PersonInfo.get_or_none((PersonInfo.person_name == alias) | (PersonInfo.nickname == alias))
        if record:
            return record.person_id

        record = PersonInfo.get_or_none(
            (PersonInfo.group_nick_name.is_null(False)) & (PersonInfo.group_nick_name.contains(alias))
        )
        return record.person_id if record else ""
    except Exception as exc:
        logger.error(f"根据别名 {alias} 获取用户ID时出错 (Peewee): {exc}")
        return ""


def is_person_known(
    person_id: str = None,
    user_id: str = None,
    platform: str = None,
    person_name: str = None,
) -> bool:
    if person_id:
        person = PersonInfo.get_or_none(PersonInfo.person_id == person_id)
        return person.is_known if person else False
    if user_id and platform:
        person = PersonInfo.get_or_none(PersonInfo.person_id == get_person_id(platform, user_id))
        return person.is_known if person else False
    if person_name:
        resolved_person_id = get_person_id_by_person_name(person_name)
        person = PersonInfo.get_or_none(PersonInfo.person_id == resolved_person_id)
        return person.is_known if person else False
    return False


def get_category_from_memory(memory_point: Union[str, MemoryPointDict, None]) -> Optional[str]:
    if isinstance(memory_point, dict):
        return memory_point.get("category") or DEFAULT_MEMORY_CATEGORY
    if isinstance(memory_point, str):
        parts = memory_point.split(":", 1)
        return parts[0].strip() if len(parts) > 1 else None
    return None


def get_weight_from_memory(memory_point: Union[str, MemoryPointDict, None]) -> float:
    if isinstance(memory_point, dict):
        return float(memory_point.get("weight", 0.0))
    if isinstance(memory_point, str):
        parts = memory_point.rsplit(":", 1)
        if len(parts) <= 1:
            return -math.inf
        try:
            return float(parts[-1].strip())
        except Exception:
            return -math.inf
    return -math.inf


def get_memory_content_from_memory(memory_point: Union[str, MemoryPointDict, None]) -> str:
    if isinstance(memory_point, dict):
        return (memory_point.get("content") or "").strip()
    if isinstance(memory_point, str):
        parts = memory_point.split(":")
        return ":".join(parts[1:-1]).strip() if len(parts) > 2 else ""
    return ""


def extract_categories_from_response(response: str) -> list[str]:
    if not isinstance(response, str):
        return []
    return re.findall(r"<([^<>]+)>", response)


def calculate_string_similarity(s1: str, s2: str) -> float:
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    distance = levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    return 1 - (distance / max_len if max_len > 0 else 0)


def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]
