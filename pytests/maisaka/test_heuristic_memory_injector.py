from __future__ import annotations

from types import SimpleNamespace
import pytest

from src.maisaka.memory import heuristic_injector as hi_module
from src.services.memory_service import MemoryHit, MemorySearchResult


def _make_msg(
    message_id: str,
    ts: float,
    user_id: str,
    nickname: str,
    text: str,
    *,
    reply_to: str = "",
    is_at: bool = False,
    is_emoji: bool = False,
    is_picture: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        message_id=message_id,
        time=ts,
        timestamp=ts,
        platform="qq",
        processed_plain_text=text,
        reply_to=reply_to,
        is_at=is_at,
        is_mentioned=is_at,
        is_emoji=is_emoji,
        is_picture=is_picture,
        raw_message=SimpleNamespace(components=[]),
        message_info=SimpleNamespace(
            user_info=SimpleNamespace(
                user_id=user_id,
                user_nickname=nickname,
                user_cardname="",
            )
        ),
    )


@pytest.mark.asyncio
async def test_heuristic_injector_recalls_cross_window_self_replies_and_focus_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bot_uid = "2782049248"
    monkeypatch.setattr(
        hi_module.global_config.a_memorix.integration,
        "heuristic_memory_recall_enabled",
        True,
    )
    monkeypatch.setattr(hi_module, "is_bot_self", lambda platform, uid: str(uid) == bot_uid)
    monkeypatch.setattr(
        hi_module.chat_manager,
        "get_existing_session_by_session_id",
        lambda sid: SimpleNamespace(
            session_id=sid,
            platform="qq",
            is_group_session=True,
            group_id="921747567",
            user_id="",
        ),
    )
    monkeypatch.setattr(hi_module.chat_manager, "get_session_name", lambda sid: "测试群")
    monkeypatch.setattr(hi_module, "count_messages", lambda **kwargs: 200)

    now = 1762914726.0  # 2026-11-12 10:32:06
    window_start = now - 3600.0

    recent_messages = [
        _make_msg(f"w{idx}", window_start + idx * 10, "u1", "奶龙", f"三角洲大红砖消息 {idx}")
        for idx in range(18)
    ] + [
        _make_msg("w18", now - 30.0, "u2", "yhggeM", "今天吃啥 @🍗", is_at=True),
        _make_msg("w19", now - 3.0, "u2", "yhggeM", "让你给我推荐 @🍗", is_at=True),
    ]

    historical_messages = [
        _make_msg("h1", now - 172800.0, "u3", "初君", "@🍗 晚上吃什么", is_at=True),
        _make_msg("h2", now - 172790.0, bot_uid, "麦麦", "来顿黄焖鸡米饭", reply_to="h1"),
        _make_msg("h3", now - 64600.0, "u4", "蛋白", "@🍗 晚上吃什么", is_at=True),
        _make_msg("h4", now - 64580.0, bot_uid, "麦麦", "吃点螺蛳粉或者黄焖鸡", reply_to="h3"),
        _make_msg("h5", now - 64500.0, "u4", "蛋白", "[回复了🍗的消息: 吃点螺蛳粉或者黄焖鸡] 怎么老是这些"),
        _make_msg("h6", now - 64495.0, "u4", "蛋白", "来点不一样的"),
        _make_msg("h7", now - 64470.0, bot_uid, "麦麦", "隆江猪脚饭 沙县小吃 东北水饺 选一个吧", reply_to="h6"),
    ]

    def fake_find_messages(**kwargs):
        if kwargs.get("before_time") is not None:
            return historical_messages
        return recent_messages

    monkeypatch.setattr(hi_module, "find_messages", fake_find_messages)

    injector = hi_module.HeuristicMemoryInjector()

    async def fake_impression(context):
        window_str = injector._format_message_window(context.recent_messages, context.focus_messages)
        assert "今天吃啥" in window_str
        assert "【当前触发焦点" in window_str
        return "大家早上聊游戏，最后yhggeM询问今天吃什么"

    monkeypatch.setattr(injector, "_build_chat_impression", fake_impression)

    async def fake_search(query: str, **kwargs):
        if "吃啥" in query or "推荐" in query:
            return MemorySearchResult(
                hits=[
                    MemoryHit(
                        content="蛋白表示不想老是吃螺蛳粉或黄焖鸡，想换点不一样的",
                        score=0.82,
                        hash_value="food_pref",
                        source="chat_summary:test_group",
                        metadata={"source_type": "chat_summary", "chat_id": "test_group"},
                    )
                ]
            )
        return MemorySearchResult(
            hits=[
                MemoryHit(
                    content="Maniac.分享了Smoggy宣布入驻B站的视频",
                    score=0.85,
                    hash_value="smoggy_vid",
                    source="chat_summary:test_group",
                    metadata={"source_type": "chat_summary", "chat_id": "test_group"},
                )
            ]
        )

    monkeypatch.setattr(hi_module.memory_service, "search", fake_search)

    result = await injector.build_injection_message(
        session_id="test_group",
        anchor_message=recent_messages[-1],
        source_messages=recent_messages[-2:],
    )

    assert "【关联长期记忆-内部参考" in result
    assert "蛋白表示不想老是吃螺蛳粉或黄焖鸡" in result
    assert "【你近期在本群的历史发言回顾-内部参考" in result
    assert "吃点螺蛳粉或者黄焖鸡" in result
    assert "怎么老是这些 / 来点不一样的" in result
    assert "隆江猪脚饭 沙县小吃 东北水饺 选一个吧" in result
    assert "来顿黄焖鸡米饭" in result
    assert "[11-10 周一 " in result
    assert "[11-11 周二 " in result
