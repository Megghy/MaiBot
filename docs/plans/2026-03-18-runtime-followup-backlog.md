# Runtime Follow-Up Backlog

## Status Update

As of 2026-03-18, the planned follow-up wave in this document has been completed.

Completed items:
- 1. Removed duplicated planner pre-build flow in `brain_chat`
- 2. Collapsed `BrainPlanner` / `ActionPlanner` onto a shared planner runtime
- 3. Moved local component disable-state into `src/runtime/`
- 4. Moved event dispatch ownership into `src/runtime/`
- 5. Unified `HeartFChatting` / `BrainChatting` controller scaffolding
- 6. Replaced decorator-based external plugin discovery with explicit `PLUGIN_CLASS`
- 7. Shrunk the broad `plugin_system` top-level facade and moved in-repo imports to narrower modules
- 8. Split startup orchestration responsibilities out of `main.py`
- 9. Expanded runtime-oriented tests around bootstrap, events, controller seams, and local state

Remaining backlog items:
- None from this document.

Former optional future work completed on 2026-03-18:
- Reduced active reliance on `src/plugin_system/core/` compatibility shells and switched remaining in-repo runtime paths to `runtime_registries` / runtime-native modules.
- Converted `src/plugin_system/apis/__init__.py` to a lazy namespace and narrowed the intentionally public API surface; kept `src/plugin_system/core/__init__.py` as a compatibility export only.
- Added precise `pending_restart` field-level classification for `maim_message` hot reload boundaries.
- Added startup/shutdown smoke coverage for `bot.py`.

---

## Summary

The main runtime rearchitecture has been completed.

What remains is no longer "must-fix to finish the migration", but "high-value cleanup and simplification" work. These items are optional follow-up refactors that can still materially improve maintainability, remove compatibility shells, and further reduce duplicated orchestration.

Scope assumption for this backlog:
- Config file compatibility: keep
- Database compatibility: keep
- Internal API / plugin API compatibility: can be broken if it simplifies the codebase

---

## P0: Highest-Value Follow-Ups

### 1. Remove duplicated planner pre-build flow in `brain_chat`

Current issue:
- `src/chat/brain_chat/brain_chat.py` still prepares planner prompt context before calling `BrainPlanner.plan()`.
- `src/chat/brain_chat/brain_planner.py` then rebuilds overlapping context again internally.

Why it is worth doing:
- Duplicate prompt/context construction means duplicated reads, duplicated branching, and more places to keep `ON_PLAN` behavior aligned.
- This is the most obvious remaining orchestration duplication in the active private-chat path.

Target outcome:
- Let planner prompt creation happen in exactly one place.
- Either:
  - move `ON_PLAN` interception fully inside planner execution, or
  - make `BrainPlanner.plan()` accept precomputed prompt context and stop rebuilding it.

Files likely involved:
- `src/chat/brain_chat/brain_chat.py`
- `src/chat/brain_chat/brain_planner.py`

### 2. Collapse `BrainPlanner` and `ActionPlanner` into one shared planner core

Current issue:
- `BrainPlanner` is now much thinner than before, but it is still a parallel planner implementation.
- Group/private planning now share tool parsing and activation filtering, but still not a single planner runtime.

Why it is worth doing:
- Planner evolution still needs to be done in two classes.
- Future policy changes, prompt changes, and fallback behavior can drift again.

Target outcome:
- Extract a shared planner base or shared planner service.
- Keep only very small mode-specific wrappers:
  - group planner policy
  - private planner policy

Files likely involved:
- `src/chat/brain_chat/brain_planner.py`
- `src/chat/planner_actions/planner.py`
- `src/chat/planner_actions/planner_prompting.py`
- `src/chat/planner_actions/planner_selection.py`

### 3. Move local component disable-state out of `plugin_system`

Current issue:
- `src/plugin_system/core/global_announcement_manager.py` is already a thin state shell, but the state still conceptually belongs to runtime/session control rather than plugin infrastructure.

Why it is worth doing:
- This is one of the last active runtime state owners still living under `plugin_system/core`.
- It keeps the mental model blurry: runtime behavior is still partially hidden under old plugin naming.

Target outcome:
- Move local enable/disable state to `src/runtime/`.
- Leave `global_announcement_manager` as a compatibility shim or remove it entirely if all call sites are updated.

Files likely involved:
- `src/plugin_system/core/global_announcement_manager.py`
- `src/plugin_system/apis/component_manage_api.py`
- `src/chat/message_receive/bot.py`
- `src/chat/planner_actions/action_modifier.py`
- `src/plugin_system/core/tool_use.py`
- `src/plugin_system/core/events_manager.py`

---

## P1: Strong Follow-Ups

### 4. Move event dispatch ownership from `plugin_system` to `runtime`

Current issue:
- `events_manager` already reads from runtime registries, but it still lives under `src/plugin_system/core/events_manager.py`.
- It still contains message transformation logic and stream lookups that are really runtime concerns.

Why it is worth doing:
- This is one of the last major "old namespace, new behavior" compatibility leftovers.
- It hides runtime behavior behind a plugin-era facade.

Target outcome:
- Introduce a runtime-native event dispatcher module.
- Make `plugin_system.core.events_manager` a compatibility export only, or remove it after updating imports.

Files likely involved:
- `src/plugin_system/core/events_manager.py`
- `src/runtime/`
- `src/main.py`
- `bot.py`

### 5. Unify `HeartFChatting` and `BrainChatting` controller scaffolding

Current issue:
- Group and private chat controllers still keep parallel loop-control, cycle tracking, action execution flow, and history bookkeeping.
- They are closer than before, but still structurally similar enough to justify one more consolidation pass.

Why it is worth doing:
- These are both active runtime controllers.
- They are large and still expensive to reason about.

Target outcome:
- Extract shared controller/runtime loop primitives.
- Keep only mode-specific decision behavior and reply policy separated.

Files likely involved:
- `src/chat/heart_flow/heartFC_chat.py`
- `src/chat/brain_chat/brain_chat.py`
- `src/runtime/session.py`

### 6. Remove decorator-based plugin-class discovery for self-use setup

Current issue:
- External plugins still rely on dynamic import plus `@register_plugin`.
- This is now compatibility-oriented rather than architecture-oriented.

Why it is worth doing:
- For a self-use project, the old plugin API may no longer be worth preserving.
- A direct manifest/module contract would be simpler than "import side effects register classes, then instantiate later".

Target outcome:
- Choose one simpler external plugin contract:
  - explicit module-level exported plugin class
  - explicit `register(runtime)` entrypoint
  - manifest + direct loader mapping
- After that, delete `plugin_register_api` and more of the old plugin bootstrap compatibility surface.

Files likely involved:
- `src/plugin_system/apis/plugin_register_api.py`
- `src/plugin_system/core/plugin_manager.py`
- external plugin packages under `plugins/`

---

## P2: Medium-Value Cleanups

### 7. Shrink `plugin_system.__init__` and old public plugin facade further

Current issue:
- `src/plugin_system/__init__.py` still exports a large public surface.
- It works, but it still preserves a broad compatibility layer.

Why it is worth doing:
- Broad compatibility facades make the codebase harder to reason about.
- For self-use, a narrower public surface is easier to maintain.

Target outcome:
- Keep only the symbols that are still intentionally public.
- Update in-repo plugins/imports to narrower source modules.

Files likely involved:
- `src/plugin_system/__init__.py`
- in-repo plugin files under `src/plugins/` and `plugins/`

### 8. Split startup responsibilities in `main.py`

Current issue:
- `src/main.py` still mixes:
  - service bootstrap
  - task registration
  - runtime startup
  - message handler wiring
  - WebUI server setup

Why it is worth doing:
- Startup is already much cleaner than before, but still centralized in one orchestration file.

Target outcome:
- Split startup into small service/bootstrap modules.
- Keep `MainSystem` as a thin top-level coordinator.

Files likely involved:
- `src/main.py`
- `src/runtime/bootstrap.py`
- possibly new `src/runtime/startup_*.py` helpers

### 9. Expand test coverage from helpers to integration paths

Current issue:
- Current tests cover the new runtime core and selected helper behavior well enough for safety, but not the full controller integration path.

Why it is worth doing:
- The next refactor wave will likely target planners/controllers/events, where regressions are more orchestration-heavy.

Target outcome:
- Add tests for:
  - runtime bootstrap loading behavior
  - event dispatch with real component registrations
  - private/group controller integration seams
  - local component enable/disable behavior

Files likely involved:
- `tests/runtime/`
- `tests/planner/`
- possible new `tests/chat/`

---

## Recommended Order

If continuing immediately, the highest-leverage order is:

1. Remove duplicated planner pre-build flow in `brain_chat`
2. Collapse `BrainPlanner` and `ActionPlanner` into one shared planner core
3. Move local disable-state into `runtime`
4. Move event dispatch ownership into `runtime`
5. Unify `HeartFChatting` and `BrainChatting` scaffolding
6. Simplify external plugin contract if compatibility is no longer desired

---

## What Is Already Good Enough

These areas do not currently block further use and do not need urgent rework:

- Runtime bootstrap is already the active startup path.
- Built-in plugin loading no longer depends on the old thick startup flow.
- Message receive path is already runtime-first.
- Session-based direct dispatch is already active.
- Core milestone verification is already in place and passing.

---

## Exit Criteria For The Next Wave

The next refactor wave can be considered complete when:

- planner logic no longer exists in two parallel forms
- active runtime state no longer lives under `plugin_system/core`
- group/private controllers share a clear common runtime scaffold
- external plugin loading is either intentionally simplified or intentionally frozen as compatibility-only
