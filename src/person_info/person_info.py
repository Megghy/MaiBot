import uuid
import hashlib
import asyncio
import json
import time
import random
import math

from json_repair import repair_json
from typing import Any, Iterable, List, Optional, TypedDict, Union, Callable

from src.common.logger import get_logger
from src.common.database.database import db
from src.common.database.database_model import PersonInfo, PersonGroupMemory
from src.llm_models.utils_model import LLMRequest
from src.config.config import global_config, model_config
from src.chat.message_receive.chat_stream import get_chat_manager
from src.common.vector_store.manager import vector_manager


logger = get_logger("person_info")

relation_selection_model = LLMRequest(
    model_set=model_config.model_task_config.utils_small, request_type="relation_selection"
)

DEFAULT_MEMORY_CATEGORY = "其他"
MAX_MEMORY_POINTS = 200
MAX_MEMORY_POINTS_PER_CATEGORY = 30
MEMORY_SIMILARITY_THRESHOLD = 0.9
MEMORY_MIN_WEIGHT = 0.05
MEMORY_MAX_WEIGHT = 5.0
MEMORY_DECAY_GRACE_SECONDS = 6 * 60 * 60  # 6 hours
MEMORY_DECAY_TIMESPAN = 14 * 24 * 60 * 60  # 14 days
MEMORY_RECENCY_TIMESPAN = 7 * 24 * 60 * 60   # 7 days
MEMORY_FORGET_THRESHOLD = 0.15
MEMORY_HARD_FORGET_SECONDS = 180 * 24 * 60 * 60  # ~6 months


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
CATEGORY_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("喜好", ("喜欢", "爱吃", "偏好", "最爱", "爱喝", "爱看")),
    ("经历", ("经历", "发生", "去了", "参加", "上次", "以前")),
    ("性格", ("性格", "脾气", "内向", "外向", "自信", "害羞")),
    ("能力", ("擅长", "会", "技能", "专业", "懂", "熟悉")),
]


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
    weight = _normalize_weight(parts[-1])
    return MemoryPointDict(
        category=category,
        content=content,
        weight=weight,
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
        source = memory_point.get("source")
        context = memory_point.get("context")
        if source:
            normalized["source"] = str(source)
        if context:
            normalized["context"] = str(context)
        return normalized
    if isinstance(memory_point, str):
        return _legacy_memory_to_dict(memory_point)
    return None


def _apply_memory_decay(memory_point: MemoryPointDict, now: Optional[float] = None) -> None:
    if not memory_point:
        return
    now = now or _current_timestamp()
    last_used = memory_point.get("last_used_at") or memory_point.get("updated_at") or memory_point.get("created_at") or now
    elapsed = max(0.0, now - last_used)
    if elapsed <= MEMORY_DECAY_GRACE_SECONDS:
        return
    decay_factor = math.exp(-elapsed / MEMORY_DECAY_TIMESPAN)
    memory_point["weight"] = max(MEMORY_MIN_WEIGHT, round(memory_point.get("weight", 1.0) * decay_factor, 4))
    memory_point["updated_at"] = now


def _should_forget_memory(memory_point: MemoryPointDict, now: Optional[float] = None) -> bool:
    now = now or _current_timestamp()
    last_used = memory_point.get("last_used_at") or memory_point.get("updated_at") or memory_point.get("created_at") or now
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
    memory_points: List[MemoryPointDict], *, predicate: Optional[Callable[[MemoryPointDict], bool]] = None
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
    """获取唯一id"""
    if "-" in platform:
        platform = platform.split("-")[1]
    components = [platform, str(user_id)]
    key = "_".join(components)
    return hashlib.md5(key.encode()).hexdigest()


def get_person_id_by_person_name(person_name: str) -> str:
    """根据用户名获取用户ID"""
    try:
        record = PersonInfo.get_or_none(PersonInfo.person_name == person_name)
        return record.person_id if record else ""
    except Exception as e:
        logger.error(f"根据用户名 {person_name} 获取用户ID时出错 (Peewee): {e}")
        return ""


def get_person_id_by_alias(alias: str) -> str:
    """根据昵称/群名片等别名获取用户ID"""
    alias = (alias or "").strip()
    if not alias:
        return ""

    try:
        record = PersonInfo.get_or_none(
            (PersonInfo.person_name == alias) | (PersonInfo.nickname == alias)
        )
        if record:
            return record.person_id

        record = PersonInfo.get_or_none(
            (PersonInfo.group_nick_name.is_null(False)) & (PersonInfo.group_nick_name.contains(alias))
        )
        return record.person_id if record else ""
    except Exception as e:
        logger.error(f"根据别名 {alias} 获取用户ID时出错 (Peewee): {e}")
        return ""


def is_person_known(person_id: str = None, user_id: str = None, platform: str = None, person_name: str = None) -> bool:  # type: ignore
    if person_id:
        person = PersonInfo.get_or_none(PersonInfo.person_id == person_id)
        return person.is_known if person else False
    elif user_id and platform:
        person_id = get_person_id(platform, user_id)
        person = PersonInfo.get_or_none(PersonInfo.person_id == person_id)
        return person.is_known if person else False
    elif person_name:
        person_id = get_person_id_by_person_name(person_name)
        person = PersonInfo.get_or_none(PersonInfo.person_id == person_id)
        return person.is_known if person else False
    else:
        return False


def get_category_from_memory(memory_point: Union[str, MemoryPointDict, None]) -> Optional[str]:
    """从记忆点中获取分类"""
    if isinstance(memory_point, dict):
        return memory_point.get("category") or DEFAULT_MEMORY_CATEGORY
    if isinstance(memory_point, str):
        parts = memory_point.split(":", 1)
        return parts[0].strip() if len(parts) > 1 else None
    return None


def get_weight_from_memory(memory_point: Union[str, MemoryPointDict, None]) -> float:
    """从记忆点中获取权重"""
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
    """从记忆点中获取记忆内容"""
    if isinstance(memory_point, dict):
        return (memory_point.get("content") or "").strip()
    if isinstance(memory_point, str):
        parts = memory_point.split(":")
        return ":".join(parts[1:-1]).strip() if len(parts) > 2 else ""
    return ""


def extract_categories_from_response(response: str) -> list[str]:
    """从response中提取所有<>包裹的内容"""
    if not isinstance(response, str):
        return []

    import re

    pattern = r"<([^<>]+)>"
    matches = re.findall(pattern, response)
    return matches


def calculate_string_similarity(s1: str, s2: str) -> float:
    """
    计算两个字符串的相似度

    Args:
        s1: 第一个字符串
        s2: 第二个字符串

    Returns:
        float: 相似度，范围0-1，1表示完全相同
    """
    if s1 == s2:
        return 1.0

    if not s1 or not s2:
        return 0.0

    # 计算Levenshtein距离

    distance = levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))

    # 计算相似度：1 - (编辑距离 / 最大长度)
    similarity = 1 - (distance / max_len if max_len > 0 else 0)
    return similarity


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    计算两个字符串的编辑距离

    Args:
        s1: 第一个字符串
        s2: 第二个字符串

    Returns:
        int: 编辑距离
    """
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


class Person:
    @classmethod
    def register_person(
        cls,
        platform: str,
        user_id: str,
        nickname: str,
        group_id: Optional[str] = None,
        group_nick_name: Optional[str] = None,
    ):
        """
        注册新用户的类方法
        必须输入 platform、user_id 和 nickname 参数

        Args:
            platform: 平台名称
            user_id: 用户ID
            nickname: 用户昵称
            group_id: 群号（可选，仅在群聊时提供）
            group_nick_name: 群昵称（可选，仅在群聊时提供）

        Returns:
            Person: 新注册的Person实例
        """
        if not platform or not user_id or not nickname:
            logger.error("注册用户失败：platform、user_id 和 nickname 都是必需参数")
            return None

        # 生成唯一的person_id
        person_id = get_person_id(platform, user_id)

        if is_person_known(person_id=person_id):
            logger.debug(f"用户 {nickname} 已存在")
            person = Person(person_id=person_id)
            # 如果是群聊，更新群昵称
            if group_id and group_nick_name:
                person.add_group_nick_name(group_id, group_nick_name)
            return person

        # 创建Person实例
        person = cls.__new__(cls)

        # 设置基本属性
        person.person_id = person_id
        person.platform = platform
        person.user_id = user_id
        person.nickname = nickname

        # 初始化默认值
        person.is_known = True  # 注册后立即标记为已认识
        person.person_name = nickname  # 使用nickname作为初始person_name
        person.name_reason = "用户注册时设置的昵称"
        person.know_times = 1
        person.know_since = time.time()
        person.last_know = time.time()
        person.memory_points = []
        person.group_nick_name = []  # 初始化群昵称列表

        # 如果是群聊，添加群昵称
        if group_id and group_nick_name:
            person.add_group_nick_name(group_id, group_nick_name)

        # 同步到数据库
        person.sync_to_database()

        logger.info(f"成功注册新用户：{person_id}，平台：{platform}，昵称：{nickname}")

        return person

    def __init__(self, platform: str = "", user_id: str = "", person_id: str = "", person_name: str = ""):
        self.user_id = ""
        self.platform = ""
        self.person_id = ""
        self.nickname = ""
        self.person_name: Optional[str] = None
        self.name_reason: Optional[str] = None
        self.know_times = 0
        self.know_since: Optional[float] = None
        self.last_know: Optional[float] = None
        self.memory_points: List[MemoryPointDict] = []
        self.group_nick_name: List[dict[str, str]] = []
        self.is_known = False
        self._group_memory_cache: dict[str, List[MemoryPointDict]] = {}

        if platform == global_config.bot.platform and user_id == global_config.bot.qq_account:
            self.is_known = True
            self.person_id = get_person_id(platform, user_id)
            self.user_id = user_id
            self.platform = platform
            self.nickname = global_config.bot.nickname
            self.person_name = global_config.bot.nickname
            logger.debug("初始化 Person: 机器人自身")
            return

        if person_id:
            self.person_id = person_id
        elif person_name:
            self.person_id = get_person_id_by_person_name(person_name)
            if not self.person_id:
                logger.warning(f"Person 初始化: 根据用户名 {person_name} 获取用户ID失败")
                return
        elif platform and user_id:
            self.person_id = get_person_id(platform, user_id)
            self.user_id = user_id
            self.platform = platform
        else:
            logger.error("Person 初始化失败，缺少必要参数")
            raise ValueError("Person 初始化失败，缺少必要参数")

        if not is_person_known(person_id=self.person_id):
            logger.debug(
                f"Person 初始化: 用户 {platform}:{user_id or 'unknown'}:{person_name or 'unknown'}:{self.person_id} 尚未认识"
            )
            self.person_name = f"未知用户{self.person_id[:4]}"
            return
            # raise ValueError(f"用户 {platform}:{user_id}:{person_name}:{person_id} 尚未认识")

        # 已认识用户, load 数据
        logger.debug(f"Person 初始化: 加载已认识用户 {self.person_id}")
        self.load_from_database()

    def del_memory(self, category: str, memory_content: str, similarity_threshold: float = 0.95):
        """
        删除指定分类和记忆内容的记忆点

        Args:
            category: 记忆分类
            memory_content: 要删除的记忆内容
            similarity_threshold: 相似度阈值，默认0.95（95%）

        Returns:
            int: 删除的记忆点数量
        """
        if not self.memory_points:
            return 0

        deleted_count = 0
        memory_points_to_keep: List[MemoryPointDict] = []

        for memory_point in self.memory_points:
            if not memory_point:
                continue
            memory_category = memory_point.get("category")
            memory_text = memory_point.get("content") or ""
            memory_id = memory_point.get("id")

            if memory_category != category:
                memory_points_to_keep.append(memory_point)
                continue

            similarity = calculate_string_similarity(memory_content, memory_text)
            if similarity >= similarity_threshold:
                deleted_count += 1
                if memory_id:
                    asyncio.create_task(vector_manager.delete([memory_id]))
                logger.debug(
                    f"删除记忆点: {_memory_to_log_string(memory_point)} (相似度: {similarity:.4f})"
                )
            else:
                memory_points_to_keep.append(memory_point)

        # 更新memory_points
        self.memory_points = memory_points_to_keep

        # 同步到数据库
        if deleted_count > 0:
            self.sync_to_database()
            logger.info(f"成功删除 {deleted_count} 个记忆点，分类: {category}")

        return deleted_count

    def _ensure_memory_points_initialized(self):
        if not isinstance(self.memory_points, list):
            self.memory_points = []
        else:
            sanitized = _sanitize_memory_points(self.memory_points)
            if sanitized != self.memory_points:
                self.memory_points = sanitized

    def _compact_memory_points(self):
        self._ensure_memory_points_initialized()

    def _remove_lowest_weight_point(self, *, predicate: Optional[Callable[[str], bool]] = None):
        if not self.memory_points:
            return

        predicate_dict: Optional[Callable[[MemoryPointDict], bool]] = None
        if predicate:
            predicate_dict = lambda point: predicate(_memory_to_log_string(point))  # type: ignore[arg-type]

        removed_point = _remove_lowest_weight_entry(
            self.memory_points,
            predicate=predicate_dict,
        )
        if removed_point:
            logger.debug(f"移除记忆点以腾出空间: {_memory_to_log_string(removed_point)}")

    def _ensure_memory_capacity(self, category: str):
        self._compact_memory_points()

        category_points = [point for point in self.memory_points if get_category_from_memory(point) == category]
        if len(category_points) >= MAX_MEMORY_POINTS_PER_CATEGORY:
            self._remove_lowest_weight_point(predicate=lambda p: get_category_from_memory(p) == category)

        if len(self.memory_points) >= MAX_MEMORY_POINTS:
            self._remove_lowest_weight_point()

    def add_memory_point(
        self,
        memory_content: str,
        *,
        category: Optional[str] = None,
        weight: float = 1.0,
        similarity_fn: Optional[Callable[[str, str], float]] = None,
        source: Optional[str] = None,
        context: Optional[str] = None,
    ) -> bool:
        if not self.is_known:
            logger.debug(f"用户 {self.person_id} 尚未被标记为已认识，无法添加记忆")
            return False

        memory_content = (memory_content or "").strip()
        if not memory_content:
            return False

        similarity_fn = similarity_fn or calculate_string_similarity
        category = _normalize_category(category or infer_memory_category(memory_content))
        weight_value = _normalize_weight(weight)

        self._compact_memory_points()

        now = _current_timestamp()
        duplicate_point: Optional[MemoryPointDict] = None
        for point in self.memory_points:
            if get_category_from_memory(point) != category:
                continue
            existing_content = get_memory_content_from_memory(point)
            similarity = similarity_fn(existing_content, memory_content)
            if similarity >= MEMORY_SIMILARITY_THRESHOLD:
                duplicate_point = point
                break
        
        # 准备向量元数据
        vector_metadata = {
            "person_id": self.person_id,
            "category": category,
            "weight": weight_value,
            "source": source,
            "context": context,
            "created_at": now,
        }

        if duplicate_point is not None:
            old_weight = duplicate_point.get("weight", 1.0)
            merged_weight = _normalize_weight(old_weight * 0.7 + weight_value * 0.3 + 0.2)
            duplicate_point["weight"] = merged_weight
            duplicate_point["content"] = memory_content
            duplicate_point["updated_at"] = now
            duplicate_point["last_used_at"] = now
            duplicate_point["hits"] = int(duplicate_point.get("hits", 0)) + 1
            if source:
                duplicate_point["source"] = source
            if context:
                duplicate_point["context"] = context
            
            # 更新向量（实际上是添加新的，因为内容可能微调）
            # 如果有旧ID，也许应该删除？但这里简化处理，只添加新的
            if category == "关系":
                asyncio.create_task(vector_manager.add_relation(memory_content, vector_metadata))
            else:
                asyncio.create_task(vector_manager.add_memory(memory_content, vector_metadata))
                
            logger.debug(f"更新已存在的记忆点: {_memory_to_log_string(duplicate_point)}")
        else:
            self._ensure_memory_capacity(category)
            new_id = str(uuid.uuid4())
            new_point: MemoryPointDict = {
                "id": new_id,
                "category": category,
                "content": memory_content,
                "weight": weight_value,
                "created_at": now,
                "updated_at": now,
                "last_used_at": now,
                "hits": 1,
            }
            if source:
                new_point["source"] = source
            if context:
                new_point["context"] = context
            self.memory_points.append(new_point)
            
            # 添加到向量库
            vector_metadata["id"] = new_id
            if category == "关系":
                asyncio.create_task(vector_manager.add_relation(memory_content, vector_metadata))
            else:
                asyncio.create_task(vector_manager.add_memory(memory_content, vector_metadata))
                
            logger.debug(f"新增记忆点: {_memory_to_log_string(new_point)}")

        self.sync_to_database()
        return True

    def get_all_category(self):
        category_list: List[str] = []
        for memory in self.memory_points:
            if not memory:
                continue
            category = get_category_from_memory(memory)
            if category and category not in category_list:
                category_list.append(category)
        return category_list

    def get_memory_list_by_category(self, category: str):
        memory_list: List[MemoryPointDict] = []
        for memory in self.memory_points:
            if not memory:
                continue
            if get_category_from_memory(memory) == category:
                memory_list.append(memory)
        return memory_list

    def get_random_memory_by_category(self, category: str, num: int = 1):
        memory_list = self.get_memory_list_by_category(category)
        if len(memory_list) <= num:
            return memory_list
        return random.sample(memory_list, num)

    def _calculate_relevance(self, memory_content: str, chat_content: str) -> float:
        """计算记忆内容与聊天内容的关联度（基于Bigram）"""
        if not memory_content or not chat_content:
            return 0.0

        # Bi-gram set generation
        def get_bigrams(text):
            return {text[i : i + 2] for i in range(len(text) - 1)}

        mem_grams = get_bigrams(memory_content)
        chat_grams = get_bigrams(chat_content)

        if not mem_grams:
            return 0.0

        common = mem_grams & chat_grams
        return len(common) / len(mem_grams)

    def get_relevant_memories_by_category(self, category: str, chat_content: str, num: int = 1):
        """获取与聊天内容最相关的记忆"""
        memory_list = self.get_memory_list_by_category(category)
        if not memory_list:
            return []

        if not chat_content:
            return self.get_random_memory_by_category(category, num)

        scored_memories = []
        now = _current_timestamp()
        for memory in memory_list:
            content = get_memory_content_from_memory(memory)
            relevance = self._calculate_relevance(content, chat_content)
            weight_bonus = get_weight_from_memory(memory) / MEMORY_MAX_WEIGHT
            last_used = memory.get("last_used_at") or memory.get("updated_at") or memory.get("created_at") or now
            recency = max(0.0, 1.0 - (now - last_used) / MEMORY_RECENCY_TIMESPAN)
            score = relevance * 0.6 + weight_bonus * 0.25 + recency * 0.15
            scored_memories.append((score, memory))

        # 按分数降序排序
        scored_memories.sort(key=lambda x: x[0], reverse=True)

        # 如果最高分数为0，说明没有相关性，回退到随机
        if scored_memories and scored_memories[0][0] == 0:
             if len(memory_list) < num:
                return memory_list
             return random.sample(memory_list, num)

        return [m[1] for m in scored_memories[:num]]

    async def _get_relevant_memories_by_category_vector(self, category: str, query_text: str, num: int = 1):
        """使用向量检索获取与查询内容最相关的记忆

        Args:
            category: 记忆分类
            query_text: 查询文本（聊天内容或信息类型描述）
            num: 需要的记忆条数

        Returns:
            List[MemoryPointDict]: 匹配到的记忆点列表
        """
        query_text = (query_text or "").strip()
        if not query_text or not self.is_known:
            # 无查询文本或用户未认识时，回退到本地随机/打分逻辑
            return self.get_relevant_memories_by_category(category, query_text, num)

        # 使用向量检索时只在当前用户且指定分类内搜索
        def _filter(meta: dict[str, Any]) -> bool:
            if meta.get("person_id") != self.person_id:
                return False
            if meta.get("category") != category:
                return False
            return True

        try:
            vector_results = await vector_manager.search_memory(query_text, k=max(num * 2, num + 1), filter_fn=_filter)
        except Exception as e:
            logger.warning(f"向量关系记忆检索失败, 使用回退逻辑: {e}")
            return self.get_relevant_memories_by_category(category, query_text, num)

        if not vector_results:
            return self.get_relevant_memories_by_category(category, query_text, num)

        # 依据相似度排序并取前 num 条
        sorted_results = sorted(vector_results, key=lambda x: float(x.get("similarity", 0.0)), reverse=True)
        top_results = sorted_results[:num]

        # 将向量结果映射回现有 memory_points（通过 id 匹配），以保持后续处理逻辑一致
        id_to_memory: dict[str, MemoryPointDict] = {}
        for m in self.memory_points:
            if not m:
                continue
            mid = str(m.get("id")) if m.get("id") is not None else None
            if mid:
                id_to_memory[mid] = m

        matched_memories: List[MemoryPointDict] = []
        for item in top_results:
            vid = str(item.get("id")) if item.get("id") is not None else None
            if vid and vid in id_to_memory:
                matched_memories.append(id_to_memory[vid])

        # 如果因为历史原因找不到对应内存结构，则回退到原逻辑
        if not matched_memories:
            return self.get_relevant_memories_by_category(category, query_text, num)

        return matched_memories

    def get_group_memory_points(self, group_id: str) -> List[MemoryPointDict]:
        group_id = (group_id or "").strip()
        if not group_id:
            return []

        cache = getattr(self, "_group_memory_cache", None)
        if cache is None:
            cache = {}
            self._group_memory_cache = cache

        if group_id in cache:
            return cache[group_id]

        memory_points: List[MemoryPointDict] = []
        try:
            record = PersonGroupMemory.get_or_none(
                (PersonGroupMemory.person_id == self.person_id) & (PersonGroupMemory.group_id == group_id)
            )
            if record and record.memory_points:
                loaded_points = json.loads(record.memory_points)
                if isinstance(loaded_points, list):
                    memory_points = _sanitize_memory_points(loaded_points)
        except Exception as e:
            logger.warning(f"解析用户 {self.person_id} 在群 {group_id} 的记忆失败: {e}")

        cache[group_id] = memory_points
        return memory_points

    def add_group_memory_point(
        self,
        group_id: str,
        memory_content: str,
        *,
        category: Optional[str] = None,
        weight: float = 1.0,
        similarity_fn: Optional[Callable[[str, str], float]] = None,
        source: Optional[str] = None,
        context: Optional[str] = None,
    ) -> bool:
        if not self.is_known:
            logger.debug(f"用户 {self.person_id} 尚未被标记为已认识，无法添加群记忆")
            return False

        group_id = (group_id or "").strip()
        if not group_id:
            return False

        memory_content = (memory_content or "").strip()
        if not memory_content:
            return False

        similarity_fn = similarity_fn or calculate_string_similarity
        category = _normalize_category(category or infer_memory_category(memory_content))
        weight_value = _normalize_weight(weight)

        memory_points = self.get_group_memory_points(group_id)

        duplicate_point: Optional[MemoryPointDict] = None
        now = _current_timestamp()
        for point in memory_points:
            if get_category_from_memory(point) != category:
                continue
            existing_content = get_memory_content_from_memory(point)
            similarity = similarity_fn(existing_content, memory_content)
            if similarity >= MEMORY_SIMILARITY_THRESHOLD:
                duplicate_point = point
                break
        
        vector_metadata = {
            "person_id": self.person_id,
            "group_id": group_id,
            "category": category,
            "weight": weight_value,
            "source": source,
            "context": context,
            "created_at": now,
            "is_group": True,
        }

        if duplicate_point is not None:
            old_weight = duplicate_point.get("weight", 1.0)
            merged_weight = _normalize_weight(old_weight * 0.7 + weight_value * 0.3 + 0.2)
            duplicate_point["weight"] = merged_weight
            duplicate_point["content"] = memory_content
            duplicate_point["updated_at"] = now
            duplicate_point["last_used_at"] = now
            duplicate_point["hits"] = int(duplicate_point.get("hits", 0)) + 1
            if source:
                duplicate_point["source"] = source
            if context:
                duplicate_point["context"] = context
            
            if category == "关系":
                asyncio.create_task(vector_manager.add_relation(memory_content, vector_metadata))
            else:
                asyncio.create_task(vector_manager.add_memory(memory_content, vector_metadata))
                
            logger.debug(f"更新已存在的群记忆点: {_memory_to_log_string(duplicate_point)}")
        else:
            category_points = [p for p in memory_points if get_category_from_memory(p) == category]
            if len(category_points) >= MAX_MEMORY_POINTS_PER_CATEGORY:
                removed_point = _remove_lowest_weight_entry(
                    memory_points, predicate=lambda p: get_category_from_memory(p) == category
                )
                if removed_point and removed_point.get("id"):
                    asyncio.create_task(vector_manager.delete([removed_point["id"]]))
                if removed_point:
                    logger.debug(f"移除群记忆点以腾出空间: {_memory_to_log_string(removed_point)}")

            if len(memory_points) >= MAX_MEMORY_POINTS:
                removed_point = _remove_lowest_weight_entry(memory_points)
                if removed_point and removed_point.get("id"):
                    asyncio.create_task(vector_manager.delete([removed_point["id"]]))
                if removed_point:
                    logger.debug(f"移除群记忆点以腾出空间: {_memory_to_log_string(removed_point)}")

            new_id = str(uuid.uuid4())
            new_point: MemoryPointDict = {
                "id": new_id,
                "category": category,
                "content": memory_content,
                "weight": weight_value,
                "created_at": now,
                "updated_at": now,
                "last_used_at": now,
                "hits": 1,
            }
            if source:
                new_point["source"] = source
            if context:
                new_point["context"] = context
            memory_points.append(new_point)
            
            vector_metadata["id"] = new_id
            if category == "关系":
                asyncio.create_task(vector_manager.add_relation(memory_content, vector_metadata))
            else:
                asyncio.create_task(vector_manager.add_memory(memory_content, vector_metadata))

            logger.debug(f"新增群记忆点: {_memory_to_log_string(new_point)}")

        try:
            record, created = PersonGroupMemory.get_or_create(
                person_id=self.person_id,
                group_id=group_id,
                defaults={"memory_points": json.dumps(memory_points, ensure_ascii=False)},
            )
            if not created:
                record.memory_points = json.dumps(memory_points, ensure_ascii=False)
                record.save()
            return True
        except Exception as e:
            logger.error(f"同步用户 {self.person_id} 在群 {group_id} 的群记忆到数据库时出错: {e}")
            return False

    def add_group_nick_name(self, group_id: str, group_nick_name: str):
        """
        添加或更新群昵称

        Args:
            group_id: 群号
            group_nick_name: 群昵称
        """
        if not group_id or not group_nick_name:
            return

        # 检查是否已存在该群号的记录
        for item in self.group_nick_name:
            if item.get("group_id") == group_id:
                # 更新现有记录
                item["group_nick_name"] = group_nick_name
                self.sync_to_database()
                logger.debug(f"更新用户 {self.person_id} 在群 {group_id} 的群昵称为 {group_nick_name}")
                return

        # 添加新记录
        self.group_nick_name.append({"group_id": group_id, "group_nick_name": group_nick_name})
        self.sync_to_database()
        logger.debug(f"添加用户 {self.person_id} 在群 {group_id} 的群昵称 {group_nick_name}")

    def load_from_database(self):
        """从数据库加载个人信息数据"""
        try:
            # 查询数据库中的记录
            record = PersonInfo.get_or_none(PersonInfo.person_id == self.person_id)

            if record:
                self.user_id = record.user_id or ""
                self.platform = record.platform or ""
                self.is_known = record.is_known or False
                self.nickname = record.nickname or ""
                self.person_name = record.person_name or self.nickname
                self.name_reason = record.name_reason or None
                self.know_times = record.know_times or 0

                # 处理points字段（JSON格式的列表）
                if record.memory_points:
                    try:
                        loaded_points = json.loads(record.memory_points)
                        if isinstance(loaded_points, list):
                            self.memory_points = _sanitize_memory_points(loaded_points)
                        else:
                            self.memory_points = []
                    except (json.JSONDecodeError, TypeError):
                        logger.warning(f"解析用户 {self.person_id} 的points字段失败，使用默认值")
                        self.memory_points = []
                else:
                    self.memory_points = [{"category": "default", "content": "", "weight": 1.0, "created_at": _current_timestamp(), "updated_at": _current_timestamp(), "last_used_at": _current_timestamp(), "hits": 0}]

                # 处理group_nick_name字段（JSON格式的列表）
                if record.group_nick_name:
                    try:
                        loaded_group_nick_names = json.loads(record.group_nick_name)
                        # 确保是列表格式
                        if isinstance(loaded_group_nick_names, list):
                            self.group_nick_name = loaded_group_nick_names
                        else:
                            self.group_nick_name = []
                    except (json.JSONDecodeError, TypeError):
                        logger.warning(f"解析用户 {self.person_id} 的group_nick_name字段失败，使用默认值")
                        self.group_nick_name = []
                else:
                    self.group_nick_name = []

                logger.debug(f"已从数据库加载用户 {self.person_id} 的信息")
            else:
                self.sync_to_database()
                logger.info(f"用户 {self.person_id} 在数据库中不存在，使用默认值并创建")

        except Exception as e:
            logger.error(f"从数据库加载用户 {self.person_id} 信息时出错: {e}")
            # 出错时保持默认值

    def sync_to_database(self):
        """将所有属性同步回数据库"""
        if not self.is_known:
            return
        try:
            # 准备数据
            data = {
                "person_id": self.person_id,
                "is_known": self.is_known,
                "platform": self.platform,
                "user_id": self.user_id,
                "nickname": self.nickname,
                "person_name": self.person_name,
                "name_reason": self.name_reason,
                "know_times": self.know_times,
                "know_since": self.know_since,
                "last_know": self.last_know,
                "memory_points": json.dumps(self.memory_points, ensure_ascii=False)
                if self.memory_points
                else json.dumps([], ensure_ascii=False),
                "group_nick_name": json.dumps(self.group_nick_name, ensure_ascii=False)
                if self.group_nick_name
                else json.dumps([], ensure_ascii=False),
            }

            # 检查记录是否存在
            record = PersonInfo.get_or_none(PersonInfo.person_id == self.person_id)

            if record:
                # 更新现有记录
                for field, value in data.items():
                    if hasattr(record, field):
                        setattr(record, field, value)
                record.save()
                logger.debug(f"已同步用户 {self.person_id} 的信息到数据库")
            else:
                # 创建新记录
                PersonInfo.create(**data)
                logger.debug(f"已创建用户 {self.person_id} 的信息到数据库")

        except Exception as e:
            logger.error(f"同步用户 {self.person_id} 信息到数据库时出错: {e}")

    async def build_relationship(self, chat_content: str = "", info_type=""):
        if not self.is_known:
            return ""
        # 构建points文本

        nickname_str = ""
        if self.person_name != self.nickname:
            nickname_str = f"(ta在{self.platform}上的昵称是{self.nickname})"

        relation_info = ""

        points_text = ""
        category_list = self.get_all_category()

        if chat_content:
            prompt = f"""当前聊天内容：
{chat_content}

分类列表：
{category_list}
**要求**：请你根据当前聊天内容，从以下分类中选择一个与聊天内容相关的分类，并用<>包裹输出，不要输出其他内容，不要输出引号或[]，严格用<>包裹：
例如:
<分类1><分类2><分类3>......
如果没有相关的分类，请输出<none>"""

            response, _ = await relation_selection_model.generate_response_async(prompt)
            # print(prompt)
            # print(response)
            category_list = extract_categories_from_response(response)
            if "none" not in category_list:
                for category in category_list:
                    relevant_memory = await self._get_relevant_memories_by_category_vector(category, chat_content, 2)
                    if relevant_memory:
                        random_memory_str = "\n".join(
                            [get_memory_content_from_memory(memory) for memory in relevant_memory]
                        )
                        points_text = f"有关 {category} 的内容：{random_memory_str}"
                        break
        elif info_type:
            prompt = f"""你需要获取用户{self.person_name}的 **{info_type}** 信息。

现有信息类别列表：
{category_list}
**要求**：请你根据**{info_type}**，从以下分类中选择一个与**{info_type}**相关的分类，并用<>包裹输出，不要输出其他内容，不要输出引号或[]，严格用<>包裹：
例如:
<分类1><分类2><分类3>......
如果没有相关的分类，请输出<none>"""
            response, _ = await relation_selection_model.generate_response_async(prompt)
            # print(prompt)
            # print(response)
            category_list = extract_categories_from_response(response)
            if "none" not in category_list:
                for category in category_list:
                    relevant_memory = await self._get_relevant_memories_by_category_vector(category, info_type, 3)
                    if relevant_memory:
                        random_memory_str = "\n".join(
                            [get_memory_content_from_memory(memory) for memory in relevant_memory]
                        )
                        points_text = f"有关 {category} 的内容：{random_memory_str}"
                        break
        else:
            for category in category_list:
                random_memory = self.get_random_memory_by_category(category, 1)[0]
                if random_memory:
                    points_text = f"有关 {category} 的内容：{get_memory_content_from_memory(random_memory)}"
                    break

        points_info = ""
        if points_text:
            points_info = f"你还记得有关{self.person_name}的内容：{points_text}"

        if not (nickname_str or points_info):
            return ""
        relation_info = f"{self.person_name}:{nickname_str}{points_info}"

        return relation_info

    async def record_relation(self, chat_context: str, source: str = None) -> str:
        """
        分析聊天上下文，生成关系摘要并记录
        """
        if not self.is_known:
            return ""
            
        # 1. 获取上下文辅助信息
        retrieval_context = await self.build_relationship(chat_content=chat_context)
        
        # 2. Generate summary using LLM
        prompt = f"""根据以下聊天内容和已有记忆，总结你与用户 {self.person_name} 的关系变化或当前状态。
        
聊天内容:
{chat_context}

{retrieval_context}

要求：
1. 简明扼要，一句话概括。
2. 侧重于态度、情感、承诺或重要事件。
3. 如果没有值得记录的关系变化，输出 "无"。
"""
        response, _ = await relation_selection_model.generate_response_async(prompt)
        summary = response.strip()
        
        if summary == "无" or not summary:
            return ""
            
        # 3. Store
        self.add_memory_point(summary, category="关系", source=source, context=chat_context[:100] if chat_context else None)
        
        return summary


class PersonInfoManager:
    def __init__(self):
        self.person_name_list = {}
        self.qv_name_llm = LLMRequest(model_set=model_config.model_task_config.utils, request_type="relation.qv_name")
        try:
            db.connect(reuse_if_open=True)
            # 设置连接池参数
            if hasattr(db, "execute_sql"):
                # 设置SQLite优化参数
                db.execute_sql("PRAGMA cache_size = -64000")  # 64MB缓存
                db.execute_sql("PRAGMA temp_store = memory")  # 临时存储在内存中
                db.execute_sql("PRAGMA mmap_size = 268435456")  # 256MB内存映射
            db.create_tables([PersonInfo], safe=True)
        except Exception as e:
            logger.error(f"数据库连接或 PersonInfo 表创建失败: {e}")

        # 初始化时读取所有person_name
        try:
            for record in PersonInfo.select(PersonInfo.person_id, PersonInfo.person_name).where(
                PersonInfo.person_name.is_null(False)
            ):
                if record.person_name:
                    self.person_name_list[record.person_id] = record.person_name
            logger.debug(f"已加载 {len(self.person_name_list)} 个用户名称 (Peewee)")
        except Exception as e:
            logger.error(f"从 Peewee 加载 person_name_list 失败: {e}")

    @staticmethod
    def _extract_json_from_text(text: str) -> dict:
        """从文本中提取JSON数据的高容错方法"""
        try:
            fixed_json = repair_json(text)
            if isinstance(fixed_json, str):
                parsed_json = json.loads(fixed_json)
            else:
                parsed_json = fixed_json

            if isinstance(parsed_json, list) and parsed_json:
                parsed_json = parsed_json[0]

            if isinstance(parsed_json, dict):
                return parsed_json

        except Exception as e:
            logger.warning(f"JSON提取失败: {e}")

        logger.warning(f"无法从文本中提取有效的JSON字典: {text}")
        logger.info(f"文本: {text}")
        return {"nickname": "", "reason": ""}

    async def _generate_unique_person_name(self, base_name: str) -> str:
        """生成唯一的 person_name，如果存在重复则添加数字后缀"""
        # 处理空昵称的情况
        if not base_name or base_name.isspace():
            base_name = "空格"

        # 检查基础名称是否已存在
        if base_name not in self.person_name_list.values():
            return base_name

        # 如果存在，添加数字后缀
        counter = 1
        while True:
            new_name = f"{base_name}[{counter}]"
            if new_name not in self.person_name_list.values():
                return new_name
            counter += 1

    async def qv_person_name(
        self, person_id: str, user_nickname: str, user_cardname: str, user_avatar: str, request: str = ""
    ):
        """给某个用户取名"""
        if not person_id:
            logger.debug("取名失败：person_id不能为空")
            return None

        person = Person(person_id=person_id)
        old_name = person.person_name
        old_reason = person.name_reason

        max_retries = 8
        current_try = 0
        existing_names_str = ""
        current_name_set = set(self.person_name_list.values())

        while current_try < max_retries:
            bot_name = global_config.bot.nickname

            qv_name_prompt = f"你是{bot_name}，一个伪装成人类的AI，你不能让别人发现这一点，"
            qv_name_prompt += f"现在你想给一个用户取一个昵称，用户的qq昵称是{user_nickname}，"
            qv_name_prompt += f"用户的qq群昵称名是{user_cardname}，"
            if user_avatar:
                qv_name_prompt += f"用户的qq头像是{user_avatar}，"
            if old_name:
                qv_name_prompt += f"你之前叫他{old_name}，是因为{old_reason}，"

            qv_name_prompt += f"\n其他取名的要求是：{request}，不要太浮夸，简短，"
            qv_name_prompt += "\n请根据以上用户信息，想想你叫他什么比较好，不要太浮夸，请最好使用用户的qq昵称或群昵称原文，可以稍作修改，优先使用原文。优先使用用户的qq昵称或者群昵称原文。"

            if existing_names_str:
                qv_name_prompt += f"\n请注意，以下名称已被你尝试过或已知存在，请避免：{existing_names_str}。\n"

            if len(current_name_set) < 50 and current_name_set:
                qv_name_prompt += f"已知的其他昵称有: {', '.join(list(current_name_set)[:10])}等。\n"

            qv_name_prompt += "请用json给出你的想法，并给出理由，示例如下："
            qv_name_prompt += """{
                "nickname": "昵称",
                "reason": "理由"
            }"""
            response, _ = await self.qv_name_llm.generate_response_async(qv_name_prompt)
            # logger.info(f"取名提示词：{qv_name_prompt}\n取名回复：{response}")
            result = self._extract_json_from_text(response)

            if not result or not result.get("nickname"):
                logger.error("生成的昵称为空或结果格式不正确，重试中...")
                current_try += 1
                continue

            generated_nickname = result["nickname"]

            is_duplicate = False
            if generated_nickname in current_name_set:
                is_duplicate = True
                logger.info(f"尝试给用户{user_nickname} {person_id} 取名，但是 {generated_nickname} 已存在，重试中...")
            else:

                def _db_check_name_exists_sync(name_to_check):
                    return PersonInfo.select().where(PersonInfo.person_name == name_to_check).exists()

                if await asyncio.to_thread(_db_check_name_exists_sync, generated_nickname):
                    is_duplicate = True
                    current_name_set.add(generated_nickname)

            if not is_duplicate:
                person.person_name = generated_nickname
                person.name_reason = result.get("reason", "未提供理由")
                person.sync_to_database()

                logger.info(
                    f"成功给用户{user_nickname} {person_id} 取名 {generated_nickname}，理由：{result.get('reason', '未提供理由')}"
                )

                self.person_name_list[person_id] = generated_nickname
                return result
            else:
                if existing_names_str:
                    existing_names_str += "、"
                existing_names_str += generated_nickname
                logger.debug(f"生成的昵称 {generated_nickname} 已存在，重试中...")
                current_try += 1

        # 如果多次尝试后仍未成功，使用唯一的 user_nickname 作为默认值
        unique_nickname = await self._generate_unique_person_name(user_nickname)
        logger.warning(f"在{max_retries}次尝试后未能生成唯一昵称，使用默认昵称 {unique_nickname}")
        person.person_name = unique_nickname
        person.name_reason = "使用用户原始昵称作为默认值"
        person.sync_to_database()
        self.person_name_list[person_id] = unique_nickname
        return {"nickname": unique_nickname, "reason": "使用用户原始昵称作为默认值"}


person_info_manager = PersonInfoManager()


async def store_person_memory_from_answer(person_name: str, memory_content: str, chat_id: str) -> None:
    """将人物信息存入person_info的memory_points

    Args:
        person_name: 人物名称
        memory_content: 记忆内容
        chat_id: 聊天ID
    """
    try:
        person_name = (person_name or "").strip()
        memory_content = (memory_content or "").strip()

        if not person_name or not memory_content:
            logger.debug("人物名称或记忆内容为空，跳过存储")
            return

        chat_stream = get_chat_manager().get_stream(chat_id)
        if not chat_stream:
            logger.warning(f"无法获取chat_stream for chat_id: {chat_id}")
            return

        platform = chat_stream.platform
        group_id: Optional[str] = None
        if getattr(chat_stream, "group_info", None):
            try:
                group_id = chat_stream.group_info.group_id  # type: ignore[attr-defined]
            except Exception:
                group_id = None

        if person_name == global_config.bot.nickname:
            person = Person(platform=global_config.bot.platform, user_id=global_config.bot.qq_account)
            person_id = person.person_id
        else:
            person_id = get_person_id_by_person_name(person_name)
            if not person_id:
                person_id = get_person_id_by_alias(person_name)

            if not person_id and chat_stream.user_info:
                user_nickname = (chat_stream.user_info.user_nickname or "").strip()
                if user_nickname and user_nickname == person_name:
                    person_id = get_person_id(platform, chat_stream.user_info.user_id)

            if not person_id:
                logger.warning(f"无法确定person_id，person_name: {person_name}, chat_id: {chat_id}")
                return

            person = Person(person_id=person_id)

        if not person.is_known:
            logger.warning(f"用户 {person_name} (person_id: {person_id}) 尚未认识，无法存储记忆")
            return

        category = infer_memory_category(memory_content)

        # 群聊与私聊的记忆作用域区分：
        # - 群聊：记忆只写入对应群的群级记忆（PersonGroupMemory），不写入全局
        # - 私聊：记忆写入全局 memory_points
        if group_id:
            added = person.add_group_memory_point(group_id, memory_content, category=category, weight=1.0)
        else:
            added = person.add_memory_point(memory_content, category=category, weight=1.0)

        if added:
            scope = f"群 {group_id}" if group_id else "全局"
            logger.info(f"成功添加人物记忆（{scope}）：{person_name} -> {memory_content}")
        else:
            logger.debug(f"未能添加人物记忆（可能重复）：{person_name} -> {memory_content}")

    except Exception as e:
        logger.error(f"存储人物记忆失败: {e}")
