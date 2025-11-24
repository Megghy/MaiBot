from __future__ import annotations

from typing import List, Optional

from src.person_info.person_info import (
    Person,
    get_category_from_memory,
    get_memory_content_from_memory,
)


def build_person_memory_block(person: Optional[Person], max_points: int = 4) -> str:
    """为当前回复目标生成记忆点提示"""

    if not person or not person.is_known:
        return ""

    memory_points = getattr(person, "memory_points", None)
    if not memory_points:
        return ""

    entries: List[str] = []
    for point in memory_points:
        if not isinstance(point, str):
            continue
        category = get_category_from_memory(point) or "记忆"
        content = get_memory_content_from_memory(point) or point
        content = content.strip()
        if not content:
            continue
        if len(content) > 60:
            content = content[:57].rstrip("，,。；; ") + "…"
        entries.append(f"- [{category}] {content}")
        if len(entries) >= max_points:
            break

    if not entries:
        return ""

    person_name = person.person_name or person.nickname or "对方"
    entries_text = "\n".join(entries)
    return (
        f"关于 {person_name} 的记忆要点：\n"
        f"{entries_text}\n"
        "仅当这些信息与当前对话相关时再引用。\n"
    )
