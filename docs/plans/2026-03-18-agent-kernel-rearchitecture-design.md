# Agent Kernel Rearchitecture Design

**Date:** 2026-03-18

**Goal:** Replace the current multi-path agent logic with a single runtime-centered agent kernel that unifies tool execution, prompt composition, skill profiles, and MCP integration.

**Status:** Approved for implementation in the current session.

---

## Background

The current backend has several independent LLM-driven execution paths:

- reply generation
- planner action selection
- memory retrieval ReAct loop
- tool executor

These paths duplicate concerns that should be shared:

- tool schema and execution
- prompt layering
- iteration budgeting and timeout handling
- tracing and logging
- routing between memory, knowledge, and normal chat flows

The codebase also maintains two distinct tool protocols:

- plugin/runtime tools built around `BaseTool`
- memory retrieval tools built around `MemoryRetrievalTool`

This duplication makes it harder to:

- add more MCP-backed capabilities cleanly
- introduce reusable skills
- reason about prompt quality and token growth
- debug why a reply used a tool, memory, or model knowledge

Because this is a personal project and compatibility is not a priority, the redesign should prefer a clean main path over long-term compatibility shims.

---

## Current Problems

### 1. Multiple Agent Loops

The project currently has separate decision loops for planner, memory retrieval, and tool usage. Each path has its own prompt entrypoint, tool selection behavior, and result parsing rules.

### 2. Duplicate Tool Abstractions

`BaseTool` and `MemoryRetrievalTool` solve the same problem in two different ways. This makes future MCP integration more expensive because MCP would otherwise become a third tool type.

### 3. Prompt Composition Drift

Prompt construction is spread across reply, planner, memory, and tool modules. Important concerns like policy, identity, context, memory, and output contracts are not expressed as explicit blocks.

### 4. Overlapping Reply Prefetch Paths

Reply preparation currently gathers multiple LLM-derived context blocks in parallel:

- tool results
- knowledge prompt info
- memory retrieval prompt

This causes duplicated work, inconsistent source selection, and unstable latency.

### 5. Weak Execution Boundaries

Several failure cases degrade into silent empty strings or generic fallback behavior, which makes it difficult to distinguish:

- model failure
- prompt build failure
- tool failure
- intentional no-op behavior

---

## Target Architecture

The redesign introduces a single runtime-owned agent layer under `src/runtime/agent/`.

### Core Modules

- `src/runtime/agent/models.py`
  - Shared dataclasses and enums for requests, results, traces, prompt blocks, and tool execution records.

- `src/runtime/agent/tool_spec.py`
  - The single tool protocol for the whole project.

- `src/runtime/agent/tool_registry.py`
  - Registration and lookup for built-in tools, plugin tools, memory tools, and MCP-backed tools.

- `src/runtime/agent/skills.py`
  - Declarative `SkillProfile` definitions.

- `src/runtime/agent/profiles.py`
  - Runtime-expanded agent profiles such as reply, planner, memory retrieval, and knowledge QA.

- `src/runtime/agent/mcp_adapter.py`
  - Converts MCP resources or commands into runtime `ToolSpec` objects.

- `src/runtime/agent/context_assembler.py`
  - Builds structured context for an agent run.

- `src/runtime/agent/prompt_composer.py`
  - Converts structured prompt blocks into final LLM inputs with a consistent ordering.

- `src/runtime/agent/runner.py`
  - The single execution engine supporting:
    - `single_pass`
    - `tool_loop`

### Prompt Block Model

Prompt composition should be based on named blocks rather than scattered string assembly. The standard block order is:

1. policy
2. identity
3. task
4. context
5. memory
6. tool observations
7. output contract

This ensures every agent path expresses intent consistently and makes skills easier to compose.

---

## Tool Model

All executable capabilities should be represented as a `ToolSpec`.

### Required Fields

- `name`
- `description`
- `params_schema`
- `executor`
- `source`
  - `builtin`
  - `plugin`
  - `memory`
  - `mcp`
- `tags`
- `enabled`
- `visible_to_profiles`
- `timeout_seconds`
- `fail_mode`

### Consequences

- `BaseTool` and `MemoryRetrievalTool` will be removed from the main execution path.
- MCP tools will not create a parallel runtime. They will be adapted into `ToolSpec`.
- Tool execution results will always return a shared structured result shape.

---

## Skill Model

Skills should be configuration-driven, not implemented as independent execution engines.

### `SkillProfile`

A skill describes how an agent run should behave:

- which model task to use
- which prompt blocks to enable
- which tool tags are allowed
- whether memory is off, light, or full
- whether MCP usage is disabled, allow-listed, or automatic
- iteration and timeout budgets
- output mode

### Initial Built-in Skills

- `reply_default`
- `planner_default`
- `memory_retrieval`
- `knowledge_qa`
- `tool_only`

These should be sufficient for the first rollout. More granular skills can be added later if they map to clear behavior differences.

---

## Execution Flow

The new runtime flow for a message is:

1. `RuntimeSession.receive(message)`
2. `SkillResolver.resolve(message, session_state)`
3. `ContextAssembler.build(request, profile)`
4. `PromptComposer.compose(context, profile)`
5. `AgentRunner.run(request, profile)`
6. `ResultInterpreter` converts `AgentResult` into planner actions, reply text, or QA output

### Important Routing Rules

- memory retrieval is no longer a mandatory reply prefetch step
- MCP use is opt-in through profile policy
- normal chat should not automatically enter a multi-iteration tool loop
- planner and reply remain distinct tasks, but share the same kernel and result model

---

## Module Migration Strategy

This redesign intentionally prefers a direct cutover over long-lived compatibility support.

### Modules to Replace as Main Paths

- `src/plugin_system/core/tool_use.py`
- `src/memory_system/retrieval_agent.py`
- `src/chat/planner_actions/planner_runtime.py`
- reply prefetch behavior inside `src/chat/replyer/base_replyer.py`

### Migration Rule

When a module is cut over to the new runtime:

- old logic should be reduced to a thin wrapper only if needed for transition
- duplicated tool schemas should be removed
- prompt templates that no longer fit the block model should be deleted or rewritten

---

## Error Handling

The new agent kernel should follow fail-fast rules:

- prompt build failures should surface as explicit agent errors
- tool execution should preserve structured failure records
- agent timeout should be explicit in `AgentResult.status`
- planner/reply should not quietly continue after critical prompt composition failures

Soft failures are acceptable only for optional external context providers when the profile explicitly allows them.

---

## Testing Strategy

### Kernel Tests

- runner execution mode tests
- tool registry filtering tests
- prompt block composition tests
- skill resolution tests

### Migration Tests

- planner action selection through the new runner
- memory retrieval tool-loop through the new runner
- reply context assembly through the new context assembler

### Integration Tests

- MCP-backed tool registration and invocation
- plugin tool registration through the unified tool registry
- end-to-end runtime message flow using a selected skill

---

## Acceptance Criteria

The redesign is considered complete when:

- there is one main tool schema in active use
- there is one main agent execution loop in active use
- reply preparation no longer triggers three independent LLM-driven prefetch chains
- planner, memory retrieval, and reply all run through the unified agent kernel
- at least one MCP-backed tool source is registered through the adapter layer
- skills are real execution profiles, not placeholder config objects
- outdated main-path code is removed or reduced to thin wrappers only
- `pytest` and `ruff check .` pass

---

## Implementation Order

1. Create the new agent kernel skeleton
2. Unify tool schema and registry
3. Migrate memory retrieval and planner
4. Migrate reply context assembly and prompt composition
5. Add skills and MCP adapters
6. Remove obsolete main-path logic and verify behavior

