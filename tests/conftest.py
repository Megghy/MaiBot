from __future__ import annotations

from types import SimpleNamespace

import pytest


SUPPORTED_LIVE_MODEL_TASKS = {
    "utils",
    "utils_small",
    "tool_use",
    "replyer",
    "reply_splitter",
    "planner",
    "expression",
    "lpmm_entity_extract",
    "lpmm_rdf_build",
    "lpmm_qa",
    "embedding",
}


class FakeMessage:
    def __init__(self, text: str = "hello") -> None:
        self.processed_plain_text = text
        self.chat_stream = None
        self.is_command = False
        self.updated_stream = None

    def update_chat_stream(self, chat_stream) -> None:
        self.chat_stream = chat_stream
        self.updated_stream = chat_stream


@pytest.fixture
def fake_chat_stream():
    return SimpleNamespace(
        stream_id="stream-1",
        group_info=None,
        user_info=SimpleNamespace(user_id="u1", user_nickname="tester", user_cardname=None, platform="qq"),
        platform="qq",
    )


@pytest.fixture
def fake_message():
    return FakeMessage()


def pytest_addoption(parser):
    group = parser.getgroup("live-model")
    group.addoption(
        "--run-live-model",
        action="store_true",
        default=False,
        help="运行会真实调用 config/model_config.toml 中模型的测试。",
    )
    group.addoption(
        "--live-model-task",
        action="append",
        default=[],
        metavar="TASK",
        help="指定要实测的 model_task_config 任务名，可重复传入。默认仅测试 replyer。",
    )
    group.addoption(
        "--live-model-all-models",
        action="store_true",
        default=False,
        help="测试指定任务下的全部模型；默认只测该任务配置中的第一个模型。",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "live_model: 调用 config/model_config.toml 中真实模型的集成测试，默认跳过。",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-live-model"):
        return

    skip_marker = pytest.mark.skip(reason="需要显式传入 --run-live-model 才会调用真实模型。")
    for item in items:
        if "live_model" in item.keywords:
            item.add_marker(skip_marker)


def pytest_generate_tests(metafunc):
    if "live_model_case" not in metafunc.fixturenames:
        if "live_replyer_model_name" not in metafunc.fixturenames:
            return
    from src.config.config import model_config

    if "live_replyer_model_name" in metafunc.fixturenames:
        replyer_models = model_config.model_task_config.replyer.model_list
        if not replyer_models:
            raise pytest.UsageError("replyer 任务没有可用模型，无法运行 live reply 场景测试。")
        selected_replyer_models = (
            replyer_models if metafunc.config.getoption("--live-model-all-models") else replyer_models[:1]
        )
        metafunc.parametrize("live_replyer_model_name", selected_replyer_models, ids=selected_replyer_models)

    if "live_model_case" not in metafunc.fixturenames:
        return

    selected_tasks = metafunc.config.getoption("--live-model-task") or ["replyer"]
    test_all_models = metafunc.config.getoption("--live-model-all-models")

    invalid_tasks = [task_name for task_name in selected_tasks if task_name not in SUPPORTED_LIVE_MODEL_TASKS]
    if invalid_tasks:
        raise pytest.UsageError(
            "不支持的 --live-model-task: "
            + ", ".join(invalid_tasks)
            + f"。当前支持: {', '.join(sorted(SUPPORTED_LIVE_MODEL_TASKS))}"
        )

    cases: list[tuple[str, str]] = []
    case_ids: list[str] = []
    for task_name in selected_tasks:
        task_config = model_config.model_task_config.get_task(task_name)
        model_names = task_config.model_list if test_all_models else task_config.model_list[:1]
        for model_name in model_names:
            cases.append((task_name, model_name))
            case_ids.append(f"{task_name}:{model_name}")

    metafunc.parametrize("live_model_case", cases, ids=case_ids)
