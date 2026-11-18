from contextlib import contextmanager
from datetime import datetime
from typing import Iterator
import asyncio
import json

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

from src.common.data_models.jargon_data_model import MaiJargon
from src.common.database.database_model import Jargon, JargonCreatedBy
import src.learners.jargon_miner as jargon_miner_module
from src.learners.jargon_miner import JARGON_INFERENCE_THRESHOLDS, JargonMiner


@pytest.fixture
def in_memory_db(monkeypatch: pytest.MonkeyPatch):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine, tables=[Jargon.__table__])

    @contextmanager
    def _get_db_session(auto_commit: bool = True) -> Iterator[Session]:
        with Session(engine, expire_on_commit=False) as session:
            yield session
            if auto_commit:
                session.commit()

    monkeypatch.setattr(jargon_miner_module, "get_db_session", _get_db_session)
    monkeypatch.setattr(
        jargon_miner_module.JargonConfigUtils,
        "resolve_jargon_group_scope",
        lambda session_id: ({session_id}, False),
    )
    return _get_db_session


def test_should_infer_meaning_uses_updated_thresholds() -> None:
    miner = JargonMiner("chat-1", "测试群")
    assert JARGON_INFERENCE_THRESHOLDS == [2, 5, 15, 50]

    jargon_c1 = Jargon(content="跑刀", count=1, last_inference_count=0, is_complete=False)
    assert miner._should_infer_meaning(jargon_c1) is False

    jargon_c2 = Jargon(content="跑刀", count=2, last_inference_count=0, is_complete=False)
    assert miner._should_infer_meaning(jargon_c2) is True

    jargon_after_c2 = Jargon(content="跑刀", count=4, last_inference_count=2, is_complete=False)
    assert miner._should_infer_meaning(jargon_after_c2) is False

    jargon_c5 = Jargon(content="跑刀", count=5, last_inference_count=2, is_complete=False)
    assert miner._should_infer_meaning(jargon_c5) is True


def test_warm_cache_from_db_loads_scoped_active_jargons(in_memory_db) -> None:
    now = datetime.now()
    with in_memory_db() as session:
        session.add(
            Jargon(
                content="巴克什",
                meaning="",
                session_id_dict=json.dumps({"chat-1": 2}),
                count=2,
                is_complete=False,
                created_timestamp=now,
                updated_timestamp=now,
            )
        )
        session.add(
            Jargon(
                content="其他群黑话",
                meaning="",
                session_id_dict=json.dumps({"chat-2": 2}),
                count=2,
                is_complete=False,
                created_timestamp=now,
                updated_timestamp=now,
            )
        )

    miner = JargonMiner("chat-1", "测试群")
    cached = miner.get_cached_jargons()
    assert "巴克什" in cached
    assert "其他群黑话" not in cached


@pytest.mark.asyncio
async def test_process_extracted_entries_merges_case_insensitively_and_counts_groups(
    in_memory_db,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inferred_ids: list[int] = []

    async def _fake_infer_by_id(self: JargonMiner, jargon_id: int) -> None:
        inferred_ids.append(jargon_id)

    class _DummyHookResult:
        aborted = False
        kwargs: dict[str, object] = {}

    class _DummyRuntimeManager:
        async def invoke_hook(self, *_args: object, **_kwargs: object) -> _DummyHookResult:
            return _DummyHookResult()

    monkeypatch.setattr(JargonMiner, "_infer_meaning_by_id", _fake_infer_by_id)
    monkeypatch.setattr(JargonMiner, "_get_runtime_manager", staticmethod(lambda: _DummyRuntimeManager()))

    miner = JargonMiner("chat-1", "测试群")
    entries = [
        {
            "content": "MT",
            "raw_content": {"[1] MT怎么说"},
            "evidence_messages": [[{"platform": "qq", "message_id": "m1"}]],
        },
        {
            "content": "mt",
            "raw_content": {"[2] 今天mt上了吗"},
            "evidence_messages": [[{"platform": "qq", "message_id": "m2"}]],
        },
    ]

    saved, updated = await miner.process_extracted_entries(entries)
    await asyncio.sleep(0)
    assert (saved, updated) == (1, 0)

    with in_memory_db(auto_commit=False) as session:
        records = session.exec(select(Jargon)).all()
    assert len(records) == 1
    assert records[0].count == 2
    assert inferred_ids == [records[0].id]


@pytest.mark.asyncio
async def test_infer_meaning_preserves_evidence_messages_on_failure(
    in_memory_db,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence_json = json.dumps([[{"platform": "qq", "message_id": "m1"}, {"platform": "qq", "message_id": "m2"}]])
    now = datetime.now()
    with in_memory_db() as session:
        record = Jargon(
            content="哈弗币",
            evidence_messages=evidence_json,
            session_id_dict=json.dumps({"chat-1": 2}),
            count=2,
            is_jargon=False,
            meaning="",
            created_by=JargonCreatedBy.AI,
            created_timestamp=now,
            updated_timestamp=now,
        )
        session.add(record)
        session.flush()
        record_id = record.id
        jargon_obj = MaiJargon.from_db_instance(record)

    monkeypatch.setattr(
        JargonMiner,
        "_load_evidence_contexts",
        lambda self, obj: (["[1] 为了点哈弗币连人都不当了"], obj.evidence_messages, False),
    )

    class _EmptyResult:
        response = ""
        output_items: list[object] = []
        model_name = "test"
        generation_attempts: list[object] = []

    async def _fake_generate(*_args: object, **_kwargs: object) -> _EmptyResult:
        return _EmptyResult()

    monkeypatch.setattr(jargon_miner_module.llm_inference, "generate_response", _fake_generate)

    class _DummyPrompt:
        def add_context(self, *_args: object, **_kwargs: object) -> None:
            pass

    async def _fake_render(*_args: object, **_kwargs: object) -> str:
        return "prompt"

    monkeypatch.setattr(jargon_miner_module.prompt_manager, "get_prompt", lambda _name: _DummyPrompt())
    monkeypatch.setattr(jargon_miner_module.prompt_manager, "render_prompt", _fake_render)

    miner = JargonMiner("chat-1", "测试群")
    await miner.infer_meaning(jargon_obj)

    with in_memory_db(auto_commit=False) as session:
        persisted = session.exec(select(Jargon).where(Jargon.id == record_id)).first()
    assert persisted is not None
    assert persisted.evidence_messages == evidence_json
