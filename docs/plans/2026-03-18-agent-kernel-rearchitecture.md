# Agent Kernel Rearchitecture Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the current fragmented agent, tool, prompt, and retrieval execution paths with a unified runtime-owned agent kernel that also supports skill profiles and MCP-backed tools.

**Architecture:** Build a new `src/runtime/agent/` layer and migrate planner, memory retrieval, reply preparation, and tool execution onto it in stages. Use a single tool schema, a single runner, structured prompt blocks, and configuration-driven skills, then connect MCP as another tool source instead of a separate runtime.

**Tech Stack:** Python 3.10+, asyncio, existing runtime/session layer, current LLM wrapper stack, pytest, Ruff

---

### Task 1: Create Agent Kernel Skeleton

**Files:**
- Create: `src/runtime/agent/__init__.py`
- Create: `src/runtime/agent/models.py`
- Create: `src/runtime/agent/tool_spec.py`
- Create: `src/runtime/agent/tool_registry.py`
- Create: `src/runtime/agent/skills.py`
- Create: `src/runtime/agent/profiles.py`
- Create: `src/runtime/agent/prompt_composer.py`
- Create: `src/runtime/agent/context_assembler.py`
- Create: `src/runtime/agent/runner.py`
- Test: `tests/runtime/test_agent_kernel.py`

**Step 1: Write the failing test**

Add a kernel smoke test that constructs a minimal profile and request, then verifies the runner can return a structured `AgentResult`.

**Step 2: Run test to verify it fails**

Run: `pytest tests/runtime/test_agent_kernel.py -v`
Expected: FAIL because the new agent runtime package does not exist.

**Step 3: Add shared models and runner shell**

Implement the minimal dataclasses and enums:

- `AgentRequest`
- `AgentResult`
- `PromptBlock`
- `ToolExecutionResult`
- `SkillProfile`
- `AgentMode`

Implement a minimal runner that supports `single_pass`.

**Step 4: Run kernel test**

Run: `pytest tests/runtime/test_agent_kernel.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/runtime/agent tests/runtime/test_agent_kernel.py
git commit -m "refactor: add agent kernel skeleton"
```

### Task 2: Unify Tool Schema and Registry

**Files:**
- Modify: `src/runtime/agent/tool_spec.py`
- Modify: `src/runtime/agent/tool_registry.py`
- Modify: `src/plugin_system/base/base_tool.py`
- Modify: `src/plugin_system/apis/tool_api.py`
- Modify: `src/memory_system/retrieval_tools/tool_registry.py`
- Test: `tests/runtime/test_agent_tool_registry.py`

**Step 1: Write the failing test**

Add a test that registers:

- a plugin-style tool
- a memory-style tool

Then assert both are exposed as the same unified `ToolSpec`.

**Step 2: Run test to verify it fails**

Run: `pytest tests/runtime/test_agent_tool_registry.py -v`
Expected: FAIL because the old registries still use incompatible protocols.

**Step 3: Implement the unified registration path**

- Add `ToolSpec` and `ToolParam`
- Add registry filtering by name, tag, and enabled state
- Add adapters from old plugin tool classes
- Add adapters from memory retrieval tool definitions

**Step 4: Reduce old tool registries to adapters**

- Keep old public helpers only if still referenced
- Stop treating `MemoryRetrievalTool` as a first-class runtime protocol

**Step 5: Run tests**

Run:
- `pytest tests/runtime/test_agent_tool_registry.py -v`
- `pytest tests/memory/test_retrieval_agent_sources.py -v`

Expected: PASS

**Step 6: Commit**

```bash
git add src/runtime/agent src/plugin_system/base/base_tool.py src/plugin_system/apis/tool_api.py src/memory_system/retrieval_tools/tool_registry.py tests/runtime/test_agent_tool_registry.py
git commit -m "refactor: unify runtime tool schema"
```

### Task 3: Migrate Memory Retrieval to Agent Runner

**Files:**
- Modify: `src/runtime/agent/profiles.py`
- Modify: `src/runtime/agent/runner.py`
- Modify: `src/memory_system/retrieval_agent.py`
- Modify: `src/memory_system/retrieval_agent_helpers.py`
- Modify: `src/memory_system/memory_retrieval.py`
- Test: `tests/memory/test_retrieval_agent.py`
- Test: `tests/memory/test_memory_retrieval.py`

**Step 1: Write or update failing tests**

Update tests so memory retrieval is expected to produce its result through the new runner contract instead of a private standalone loop contract.

**Step 2: Run tests to verify failure**

Run:
- `pytest tests/memory/test_retrieval_agent.py -v`
- `pytest tests/memory/test_memory_retrieval.py -v`

Expected: FAIL because the new runner-backed profile does not yet exist.

**Step 3: Implement `memory_retrieval` profile**

- Allow multi-iteration tool loop
- Restrict tools to memory and retrieval tags
- Add explicit final-answer output contract

**Step 4: Rewire retrieval execution**

- Make `retrieval_agent.py` build an agent request and call the unified runner
- Preserve existing answer formatting where still useful
- Remove duplicate loop logic when no longer needed

**Step 5: Run tests**

Run:
- `pytest tests/memory/test_retrieval_agent.py -v`
- `pytest tests/memory/test_memory_retrieval.py -v`

Expected: PASS

**Step 6: Commit**

```bash
git add src/runtime/agent src/memory_system/retrieval_agent.py src/memory_system/retrieval_agent_helpers.py src/memory_system/memory_retrieval.py tests/memory/test_retrieval_agent.py tests/memory/test_memory_retrieval.py
git commit -m "refactor: move memory retrieval to agent runner"
```

### Task 4: Migrate Planner to Agent Runner

**Files:**
- Modify: `src/runtime/agent/profiles.py`
- Modify: `src/chat/planner_actions/planner_runtime.py`
- Modify: `src/chat/planner_actions/planner_tools.py`
- Modify: `src/chat/planner_actions/planner_prompting.py`
- Modify: `src/chat/planner_actions/planner.py`
- Test: `tests/planner/test_planner_runtime.py`
- Test: `tests/planner/test_planner_helpers.py`

**Step 1: Write or update the failing test**

Add a planner test that expects the planner path to go through a unified runner result and still return parsed actions.

**Step 2: Run planner tests**

Run:
- `pytest tests/planner/test_planner_runtime.py -v`
- `pytest tests/planner/test_planner_helpers.py -v`

Expected: FAIL because the planner is still using the old direct LLM path.

**Step 3: Implement `planner` profile**

- `single_pass` agent mode
- action-selection output mode
- planner-safe tool allow-list

**Step 4: Rewire planner runtime**

- Compose planner prompt blocks through the new prompt composer
- Execute through the new runner
- Keep domain parsing of actions isolated in planner modules

**Step 5: Run planner tests**

Run:
- `pytest tests/planner/test_planner_runtime.py -v`
- `pytest tests/planner/test_planner_helpers.py -v`

Expected: PASS

**Step 6: Commit**

```bash
git add src/runtime/agent src/chat/planner_actions tests/planner/test_planner_runtime.py tests/planner/test_planner_helpers.py
git commit -m "refactor: move planner onto agent runner"
```

### Task 5: Replace Reply Prefetch Fan-Out with Context Assembler

**Files:**
- Modify: `src/runtime/agent/context_assembler.py`
- Modify: `src/runtime/agent/prompt_composer.py`
- Modify: `src/runtime/agent/profiles.py`
- Modify: `src/chat/replyer/base_replyer.py`
- Modify: `src/chat/replyer/group_generator.py`
- Modify: `src/chat/replyer/private_generator.py`
- Modify: `src/chat/replyer/prompt/replyer_prompt.py`
- Modify: `src/chat/replyer/replyer_helpers.py`
- Test: `tests/runtime/test_agent_context_assembler.py`

**Step 1: Write the failing test**

Add a test that verifies normal reply preparation no longer calls three independent LLM-driven prefetch stages for tool, knowledge, and memory at once.

**Step 2: Run the test**

Run: `pytest tests/runtime/test_agent_context_assembler.py -v`
Expected: FAIL because reply preparation still fans out through old prefetch paths.

**Step 3: Implement context assembly**

- Build named context blocks
- Route optional memory and knowledge assembly based on profile policy
- Keep lightweight chat and identity assembly in the reply path

**Step 4: Rewrite reply prompt assembly**

- Move reply prompt composition to structured blocks
- Simplify `replyer_prompt.py`
- Remove redundant prefetch-only prompt behaviors

**Step 5: Run tests**

Run:
- `pytest tests/runtime/test_agent_context_assembler.py -v`
- `pytest tests/planner -q`
- `pytest tests/memory -q`

Expected: PASS

**Step 6: Commit**

```bash
git add src/runtime/agent src/chat/replyer tests/runtime/test_agent_context_assembler.py
git commit -m "refactor: replace reply prefetch fan-out"
```

### Task 6: Add Skill Profiles and MCP Adapter

**Files:**
- Modify: `src/runtime/agent/skills.py`
- Modify: `src/runtime/agent/profiles.py`
- Modify: `src/runtime/agent/tool_registry.py`
- Modify: `src/runtime/agent/mcp_adapter.py`
- Modify: `src/runtime/__init__.py`
- Modify: `src/main.py`
- Test: `tests/runtime/test_agent_skills.py`
- Test: `tests/runtime/test_agent_mcp_adapter.py`

**Step 1: Write the failing tests**

Add tests that verify:

- skills can filter tool visibility and execution mode
- MCP adapter registers tools into the same registry as built-in tools

**Step 2: Run the tests**

Run:
- `pytest tests/runtime/test_agent_skills.py -v`
- `pytest tests/runtime/test_agent_mcp_adapter.py -v`

Expected: FAIL because skills and MCP adapter are not wired into startup.

**Step 3: Implement built-in skills**

Create at least these skills:

- `reply_default`
- `planner_default`
- `memory_retrieval`
- `knowledge_qa`

**Step 4: Implement MCP tool adaptation**

- define MCP config shape
- create tool specs from MCP providers
- load them during runtime bootstrap

**Step 5: Run tests**

Run:
- `pytest tests/runtime/test_agent_skills.py -v`
- `pytest tests/runtime/test_agent_mcp_adapter.py -v`

Expected: PASS

**Step 6: Commit**

```bash
git add src/runtime/agent src/main.py tests/runtime/test_agent_skills.py tests/runtime/test_agent_mcp_adapter.py
git commit -m "feat: add skill profiles and mcp adapter"
```

### Task 7: Remove Obsolete Main-Path Logic and Verify End-to-End

**Files:**
- Modify: `src/plugin_system/core/tool_use.py`
- Modify: `src/memory_system/retrieval_agent.py`
- Modify: `src/chat/planner_actions/planner_runtime.py`
- Modify: `src/chat/replyer/base_replyer.py`
- Test: existing runtime, planner, and memory tests as needed

**Step 1: Remove or reduce obsolete paths**

- eliminate standalone main-path tool loop behavior in `tool_use.py`
- remove duplicate retrieval loop code no longer used
- ensure the old modules delegate or disappear cleanly

**Step 2: Run targeted suites**

Run:
- `pytest tests/runtime -q`
- `pytest tests/planner -q`
- `pytest tests/memory -q`

Expected: PASS

**Step 3: Run repo-wide checks**

Run:
- `python -m pytest -q`
- `python -m ruff check .`

Expected: PASS

**Step 4: Optional smoke boot**

Run: `python -u bot.py`
Expected: process starts without immediate crash; if environment-specific blockers appear, document them explicitly.

**Step 5: Commit**

```bash
git add src tests
git commit -m "refactor: remove obsolete agent execution paths"
```

