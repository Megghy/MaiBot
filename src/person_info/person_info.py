import time
from typing import List, Optional

from src.common.logger import get_logger
from src.config.config import global_config
from src.person_info.memory_ops import (
    add_group_memory_point,
    add_group_nick_name,
    add_memory_point,
    async_add_group_memory_point,
    async_add_memory_point,
    calculate_relevance,
    compact_memory_points,
    delete_memory,
    ensure_memory_capacity,
    ensure_memory_points_initialized,
    get_all_category,
    get_group_memory_points,
    get_memory_list_by_category,
    get_random_memory_by_category,
    get_relevant_memories_by_category,
    get_relevant_memories_by_category_vector,
    remove_lowest_weight_point,
)
from src.person_info.memory_schema import (
    MemoryPointDict,
    calculate_string_similarity,
    get_category_from_memory,
    get_memory_content_from_memory,
    get_person_id,
    get_person_id_by_alias,
    get_person_id_by_person_name,
    get_weight_from_memory,
    infer_memory_category,
    is_person_known,
)
from src.person_info.person_persistence import load_person_from_database, sync_person_to_database
from src.person_info.person_services import PersonInfoManager, person_info_manager, store_person_memory_from_answer
from src.person_info.relationship_ops import (
    build_relationship,
    get_relation_status_dict,
    get_relation_status_summary,
    record_relation,
)

logger = get_logger("person_info")


class Person:
    del_memory = delete_memory
    _ensure_memory_points_initialized = ensure_memory_points_initialized
    _compact_memory_points = compact_memory_points
    _remove_lowest_weight_point = remove_lowest_weight_point
    _ensure_memory_capacity = ensure_memory_capacity
    add_memory_point = add_memory_point
    async_add_memory_point = async_add_memory_point
    get_all_category = get_all_category
    get_memory_list_by_category = get_memory_list_by_category
    get_random_memory_by_category = get_random_memory_by_category
    _calculate_relevance = calculate_relevance
    get_relevant_memories_by_category = get_relevant_memories_by_category
    _get_relevant_memories_by_category_vector = get_relevant_memories_by_category_vector
    get_group_memory_points = get_group_memory_points
    add_group_memory_point = add_group_memory_point
    async_add_group_memory_point = async_add_group_memory_point
    add_group_nick_name = add_group_nick_name
    load_from_database = load_person_from_database
    sync_to_database = sync_person_to_database
    _get_relation_status_dict = get_relation_status_dict
    _get_relation_status_summary = get_relation_status_summary
    build_relationship = build_relationship
    record_relation = record_relation

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

        person_id = get_person_id(platform, user_id)
        if is_person_known(person_id=person_id):
            logger.debug(f"用户 {nickname} 已存在")
            person = Person(person_id=person_id)
            if group_id and group_nick_name:
                person.add_group_nick_name(group_id, group_nick_name)
            return person

        person = cls.__new__(cls)
        person.person_id = person_id
        person.platform = platform
        person.user_id = user_id
        person.nickname = nickname
        person.is_known = True
        person.person_name = nickname
        person.name_reason = "用户注册时设置的昵称"
        person.know_times = 1
        now = time.time()
        person.know_since = now
        person.last_know = now
        person.memory_points = []
        person.relation_status = None
        person.group_nick_name = []
        person._group_memory_cache = {}

        if group_id and group_nick_name:
            person.add_group_nick_name(group_id, group_nick_name)

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
        self.relation_status: Optional[str] = None
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

        logger.debug(f"Person 初始化: 加载已认识用户 {self.person_id}")
        self.load_from_database()


__all__ = [
    "MemoryPointDict",
    "Person",
    "PersonInfoManager",
    "calculate_string_similarity",
    "get_category_from_memory",
    "get_memory_content_from_memory",
    "get_person_id",
    "get_person_id_by_alias",
    "get_person_id_by_person_name",
    "get_weight_from_memory",
    "infer_memory_category",
    "is_person_known",
    "person_info_manager",
    "store_person_memory_from_answer",
]
