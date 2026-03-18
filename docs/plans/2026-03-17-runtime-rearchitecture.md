# Runtime Rearchitecture Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the current multi-manager chat runtime and thick plugin framework with a single in-memory runtime, thin registries/event bus, and a minimal pytest-based safety net.

**Architecture:** Build a new `src/runtime` core and make it the active execution path early. Keep config/database compatibility, but freely simplify internal APIs and built-in registration. Migrate entrypoint dispatch to the new runtime, then retire or bypass old framework layers.

**Tech Stack:** Python 3.10+, asyncio, Peewee, FastAPI/uvicorn, pytest, pytest-asyncio

## Status

As of 2026-03-18, Tasks 1-8 in this implementation plan have been completed on the active branch.

Completed result summary:
- Thin runtime core under `src/runtime/` is now the active execution path.
- Receive-side orchestration now goes through `src/runtime/message_pipeline.py`.
- Built-in plugins are registered by runtime bootstrap instead of the old thick startup path.
- `plugin_manager` is now a compatibility loader instead of the primary runtime owner.
- Session-based direct dispatch replaced the old DB-polling-driven active path.
- Memory retrieval / person info / planner / heart flow large-file splits were completed.
- A minimal pytest safety net exists and currently covers runtime plus selected planner / memory / person info behavior.
- Verification for this milestone has been run successfully:
  - `python -m pytest -q`
  - `python -m ruff check .`
  - `python -m ruff format --check .`
  - `python -u bot.py` smoke startup

For optional next-stage refactors after this milestone, see:
- `docs/plans/2026-03-18-runtime-followup-backlog.md`

---

### Task 1: Add Test Harness

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/conftest.py`
- Create: `tests/runtime/test_event_bus.py`
- Create: `tests/runtime/test_runtime_manager.py`

**Step 1: Add pytest dependencies and config**

Update `pyproject.toml` to add `pytest` and `pytest-asyncio`, plus a `tool.pytest.ini_options` section.

**Step 2: Create shared test helpers**

Add `tests/conftest.py` with fixtures for fake messages, fake chat stream data, and async test defaults.

**Step 3: Add event bus tests**

Create `tests/runtime/test_event_bus.py` covering handler ordering, interception, and message mutation.

**Step 4: Add runtime manager tests**

Create `tests/runtime/test_runtime_manager.py` covering session creation, session reuse, and direct dispatch behavior.

### Task 2: Build Thin Runtime Core

**Files:**
- Create: `src/runtime/__init__.py`
- Create: `src/runtime/models.py`
- Create: `src/runtime/event_bus.py`
- Create: `src/runtime/registries.py`
- Create: `src/runtime/session.py`
- Create: `src/runtime/manager.py`

**Step 1: Create runtime data models**

Define session/event/command/action/tool model types that are small and explicit.

**Step 2: Implement event bus**

Provide async event dispatch with ordered handlers and optional interception.

**Step 3: Implement registries**

Add direct registries for commands, actions, tools, and event handlers.

**Step 4: Implement session runtime**

Create `SessionRuntime` as the single owner of per-session runtime state.

**Step 5: Implement runtime manager**

Create `RuntimeManager` to resolve/create sessions and route incoming messages.

### Task 3: Build Active Message Pipeline

**Files:**
- Create: `src/runtime/message_pipeline.py`
- Modify: `src/chat/message_receive/bot.py`
- Modify: `src/chat/message_receive/chat_stream.py`
- Modify: `src/chat/message_receive/storage.py`

**Step 1: Create runtime-facing message pipeline**

Move direct receive-side orchestration into `src/runtime/message_pipeline.py`.

**Step 2: Route `ChatBot.message_process` into runtime**

Keep normalization and boundary checks, but delegate active work to `RuntimeManager`.

**Step 3: Simplify chat stream ownership**

Make `ChatManager` a storage-backed stream resolver instead of an active runtime owner.

**Step 4: Keep persistence as audit storage only**

Use `MessageStorage` for durable logging, not queue semantics.

### Task 4: Replace Thick Plugin Runtime

**Files:**
- Modify: `src/plugin_system/__init__.py`
- Modify: `src/plugin_system/core/plugin_manager.py`
- Modify: `src/plugin_system/core/component_registry.py`
- Modify: `src/plugin_system/core/events_manager.py`
- Modify: `src/plugin_system/core/global_announcement_manager.py`
- Modify: `src/plugin_system/apis/plugin_register_api.py`
- Modify: built-in plugin files under `src/plugins/built_in/`

**Step 1: Rewire active exports**

Make `plugin_system` public exports point to the thin runtime registries/event bus.

**Step 2: Migrate built-ins to direct registration**

Replace plugin-class-driven registration with explicit direct registration during startup.

**Step 3: Collapse old manager responsibilities**

Either remove dead responsibilities or reduce old managers to compatibility shells delegating to runtime registries.

### Task 5: Integrate Reply and Decision Flow

**Files:**
- Modify: `src/chat/heart_flow/heartflow.py`
- Modify: `src/chat/heart_flow/heartFC_chat.py`
- Modify: `src/chat/replyer/replyer_manager.py`
- Modify: `src/chat/replyer/base_replyer.py`
- Modify: `src/chat/planner_actions/planner.py`

**Step 1: Remove DB-polling-driven execution from active path**

Use direct session-triggered observation/decision rather than reading fresh messages back from DB.

**Step 2: Move per-session state under `SessionRuntime`**

Make replyers/planners/session loop state resolve from runtime state, not separate caches.

**Step 3: Keep old heavy logic behind smaller runtime boundaries**

Retain behavior where possible, but stop letting these modules own top-level orchestration.

### Task 6: Restructure Memory Integration

**Files:**
- Modify: `src/memory_system/memory_retrieval.py`
- Modify: `src/memory_system/memory_writer.py`
- Create: `src/memory_system/retrieval_pipeline.py`
- Create: `src/memory_system/retrieval_cache.py`
- Create: `src/memory_system/retrieval_format.py`

**Step 1: Split retrieval phases**

Extract question generation, retrieval execution, cache reuse, and formatting into smaller modules.

**Step 2: Make runtime integration explicit**

Keep reply integration through one clear runtime-facing entrypoint.

**Step 3: Tighten write failure behavior**

Ensure memory write boundaries fail clearly instead of silently masking partial failures.

### Task 7: Rewire Startup

**Files:**
- Modify: `src/main.py`
- Modify: `bot.py`

**Step 1: Initialize runtime registries and built-ins at startup**

Create a single runtime boot path and register built-ins directly.

**Step 2: Keep shutdown behavior aligned with new runtime**

Make graceful shutdown stop runtime-owned tasks first, then persistence/web layers.

### Task 8: Verify

**Files:**
- No direct code changes expected

**Step 1: Run tests**

Run: `pytest -q`

**Step 2: Run Ruff**

Run: `ruff check .`

**Step 3: Run formatting check**

Run: `ruff format --check .`

**Step 4: Run startup smoke check if feasible**

Run: `python bot.py`

If full startup is too environment-heavy, document the exact blocker and the checks that did run.
