"""Common helpers for memory retrieval tools."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional


def truncate_text(text: Any, limit: int = 200) -> str:
    """Return a trimmed string representation with ellipsis."""

    if text is None:
        return ""

    text_str = str(text).strip()
    if not text_str:
        return text_str

    if len(text_str) <= limit:
        return text_str

    ellipsis = "..."
    keep = max(0, limit - len(ellipsis))
    return text_str[:keep] + ellipsis


def format_tool_response(success: bool, message: str = "", data: Optional[Dict[str, Any]] = None) -> str:
    """Serialize a standard tool response payload as JSON."""

    payload: Dict[str, Any] = {
        "success": bool(success),
        "message": message or ("查询成功" if success else ""),
        "data": data or {},
    }

    try:
        return json.dumps(payload, ensure_ascii=False)
    except TypeError:
        # Fallback to best-effort serialization
        safe_data = json.loads(json.dumps(payload["data"], default=str, ensure_ascii=False))
        payload["data"] = safe_data
        return json.dumps(payload, ensure_ascii=False)
