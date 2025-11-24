"""
根据person_name查询用户信息 - 工具实现
支持模糊查询，可以查询某个用户的所有信息
"""

import json
from datetime import datetime
from typing import Any, Dict, List
from src.common.logger import get_logger
from src.common.database.database_model import PersonInfo
from .tool_registry import register_memory_retrieval_tool
from .tool_utils import format_tool_response, truncate_text

logger = get_logger("memory_retrieval_tools")


def _format_group_nick_names(group_nick_name_field) -> List[Dict[str, str]]:
    """格式化群昵称信息

    Args:
        group_nick_name_field: 群昵称字段（可能是字符串JSON或None）

    Returns:
        str: 格式化后的群昵称信息字符串
    """
    if not group_nick_name_field:
        return []

    try:
        # 解析JSON格式的群昵称列表
        group_nick_names_data = (
            json.loads(group_nick_name_field) if isinstance(group_nick_name_field, str) else group_nick_name_field
        )

        if not isinstance(group_nick_names_data, list) or not group_nick_names_data:
            return []

        group_nick_list: List[Dict[str, str]] = []
        for item in group_nick_names_data:
            if isinstance(item, dict):
                group_id = item.get("group_id", "未知群号")
                group_nick_name = item.get("group_nick_name", "未知群昵称")
                group_nick_list.append({"group_id": str(group_id), "group_nick_name": group_nick_name})
            elif isinstance(item, str):
                group_nick_list.append({"group_id": "", "group_nick_name": item})

        return group_nick_list
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        logger.warning(f"解析群昵称信息失败: {e}")
        if isinstance(group_nick_name_field, str):
            preview = truncate_text(group_nick_name_field, 200)
            return [{"group_id": "", "group_nick_name": preview, "raw": True}]
        return []


def _build_record_dict(record: PersonInfo, match_type: str) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "match_type": match_type,
        "person_name": record.person_name or "",
        "nickname": record.nickname or "",
        "person_id": record.person_id or "",
        "platform": record.platform or "",
        "user_id": record.user_id or "",
        "is_known": bool(record.is_known),
        "know_times": int(record.know_times or 0),
    }

    if record.name_reason:
        payload["name_reason"] = record.name_reason

    if record.know_since:
        payload["know_since"] = datetime.fromtimestamp(record.know_since).strftime("%Y-%m-%d %H:%M:%S")
    if record.last_know:
        payload["last_know"] = datetime.fromtimestamp(record.last_know).strftime("%Y-%m-%d %H:%M:%S")

    group_nicks = _format_group_nick_names(getattr(record, "group_nick_name", None))
    if group_nicks:
        payload["group_nick_names"] = group_nicks

    memory_entries: List[Dict[str, Any]] = []
    if record.memory_points:
        try:
            memory_points_data = (
                json.loads(record.memory_points) if isinstance(record.memory_points, str) else record.memory_points
            )
            if isinstance(memory_points_data, list):
                for memory_point in memory_points_data:
                    if not memory_point or not isinstance(memory_point, str):
                        continue
                    parts = memory_point.split(":", 2)
                    if len(parts) >= 3:
                        memory_entries.append(
                            {
                                "category": parts[0].strip(),
                                "content": parts[1].strip(),
                                "weight": parts[2].strip(),
                            }
                        )
                    else:
                        memory_entries.append({"raw": memory_point})
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.warning(f"解析用户 {record.person_id} 的memory_points失败: {exc}")
            memory_entries.append({"raw": truncate_text(record.memory_points, 200)})

    if memory_entries:
        payload["memory_points"] = memory_entries

    return payload


async def query_person_info(person_name: str) -> str:
    """根据person_name查询用户信息，使用模糊查询

    Args:
        person_name: 用户名称（person_name字段）

    Returns:
        str: 查询结果，包含用户的所有信息
    """
    try:
        person_name = str(person_name).strip()
        if not person_name:
            return format_tool_response(False, "用户名称为空")

        # 构建查询条件（使用模糊查询）
        query = PersonInfo.select().where(PersonInfo.person_name.contains(person_name))

        # 执行查询
        records = list(query.limit(20))  # 最多返回20条记录

        if not records:
            return format_tool_response(False, f"未找到模糊匹配'{person_name}'的用户信息", {"person_name": person_name})

        # 区分精确匹配和模糊匹配的结果
        exact_matches = []
        fuzzy_matches = []

        for record in records:
            # 检查是否是精确匹配
            if record.person_name and record.person_name.strip() == person_name:
                exact_matches.append(record)
            else:
                fuzzy_matches.append(record)

        record_payloads = [_build_record_dict(record, "exact") for record in exact_matches]
        record_payloads.extend(_build_record_dict(record, "fuzzy") for record in fuzzy_matches)

        total_count = len(records)
        extra_info: Dict[str, Any] = {
            "person_name": person_name,
            "total": total_count,
            "exact_matches": len(exact_matches),
            "fuzzy_matches": len(fuzzy_matches),
            "records": record_payloads,
        }

        if total_count >= 20:
            extra_info["truncated"] = True

        return format_tool_response(True, f"找到 {total_count} 条匹配的用户信息", extra_info)

    except Exception as e:
        logger.error(f"查询用户信息失败: {e}")
        return format_tool_response(False, f"查询失败: {str(e)}")


def register_tool():
    """注册工具"""
    register_memory_retrieval_tool(
        name="query_person_info",
        description="查询某个用户的静态档案信息，如名称、昵称、平台、用户ID、qq号、群昵称以及已登记的人格/标签等。不用于推断具体聊天事件或关系细节，仅用于补全人物背景。",
        parameters=[
            {"name": "person_name", "type": "string", "description": "用户名称，用于查询用户信息", "required": True}
        ],
        execute_func=query_person_info,
    )
