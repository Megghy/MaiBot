import json

from src.common.database.database_model import PersonInfo
from src.common.logger import get_logger
from src.person_info.memory_schema import _current_timestamp, _sanitize_memory_points

logger = get_logger("person_info")


def load_person_from_database(person) -> None:
    """从数据库加载个人信息数据"""
    try:
        record = PersonInfo.get_or_none(PersonInfo.person_id == person.person_id)
        if record:
            person.user_id = record.user_id or ""
            person.platform = record.platform or ""
            person.is_known = record.is_known or False
            person.nickname = record.nickname or ""
            person.person_name = record.person_name or person.nickname
            person.name_reason = record.name_reason or None
            person.know_times = record.know_times or 0
            person.relation_status = getattr(record, "relation_status", None)

            if record.memory_points:
                try:
                    loaded_points = json.loads(record.memory_points)
                    person.memory_points = (
                        _sanitize_memory_points(loaded_points) if isinstance(loaded_points, list) else []
                    )
                except (json.JSONDecodeError, TypeError):
                    logger.warning(f"解析用户 {person.person_id} 的points字段失败，使用默认值")
                    person.memory_points = []
            else:
                person.memory_points = []

            if record.group_nick_name:
                try:
                    loaded_group_nick_names = json.loads(record.group_nick_name)
                    person.group_nick_name = (
                        loaded_group_nick_names if isinstance(loaded_group_nick_names, list) else []
                    )
                except (json.JSONDecodeError, TypeError):
                    logger.warning(f"解析用户 {person.person_id} 的group_nick_name字段失败，使用默认值")
                    person.group_nick_name = []
            else:
                person.group_nick_name = []

            logger.debug(f"已从数据库加载用户 {person.person_id} 的信息")
            return

        sync_person_to_database(person)
        logger.info(f"用户 {person.person_id} 在数据库中不存在，使用默认值并创建")
    except Exception as exc:
        logger.error(f"从数据库加载用户 {person.person_id} 信息时出错: {exc}")


def sync_person_to_database(person) -> None:
    """将所有属性同步回数据库"""
    if not person.is_known:
        return

    try:
        data = {
            "person_id": person.person_id,
            "is_known": person.is_known,
            "platform": person.platform,
            "user_id": person.user_id,
            "nickname": person.nickname,
            "person_name": person.person_name,
            "name_reason": person.name_reason,
            "know_times": person.know_times,
            "know_since": person.know_since,
            "last_know": person.last_know,
            "memory_points": json.dumps(person.memory_points or [], ensure_ascii=False),
            "relation_status": person.relation_status,
            "group_nick_name": json.dumps(person.group_nick_name or [], ensure_ascii=False),
        }

        record = PersonInfo.get_or_none(PersonInfo.person_id == person.person_id)
        if record:
            for field, value in data.items():
                if hasattr(record, field):
                    setattr(record, field, value)
            record.save()
            logger.debug(f"已同步用户 {person.person_id} 的信息到数据库")
        else:
            if not data.get("know_since"):
                data["know_since"] = _current_timestamp()
            if not data.get("last_know"):
                data["last_know"] = _current_timestamp()
            PersonInfo.create(**data)
            logger.debug(f"已创建用户 {person.person_id} 的信息到数据库")
    except Exception as exc:
        logger.error(f"同步用户 {person.person_id} 信息到数据库时出错: {exc}")
