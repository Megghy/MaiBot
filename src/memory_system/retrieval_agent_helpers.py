from __future__ import annotations


def _append_reasoning_summary(collected_info: str, reasoning_content: str, response: str, append_collected_info) -> str:
    thought_summary = reasoning_content or (response[:200] if response else "")
    if not thought_summary:
        return collected_info
    return append_collected_info(collected_info, f"[思考] {thought_summary}")


__all__ = ["_append_reasoning_summary"]
