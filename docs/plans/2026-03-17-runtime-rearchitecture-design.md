# Runtime Rearchitecture Design

**Goal:** Replace the current multi-manager, DB-loop-driven runtime with a single in-memory session runtime, a thin extension system, and a minimal test harness while keeping configuration files and database schema compatible.

## Context

The current architecture spreads conversation state across multiple managers and caches, routes incoming messages through persistence before decision making, and relies on a heavy plugin framework that is tightly coupled back into chat internals. The main pain points are:

- duplicated session state in `ChatManager`, `Heartflow`, `ReplyerManager`, and other globals
- incoming message flow that persists to `Messages` and later polls the database back for decision making
- oversized orchestration classes (`HeartFChatting`, `ActionPlanner`, `BaseReplyer`, `Person`, `memory_retrieval.py`)
- a thick plugin system (`PluginManager`, `ComponentRegistry`, `EventsManager`, `apis/*`) whose abstraction cost is higher than its current benefit

## Approved Direction

Use the most aggressive path and rebuild the runtime around a single `SessionRuntime` model. Backward compatibility is required only for configuration files and database schema. Internal APIs, plugin APIs, and built-in component wiring may change freely.

## Target Architecture

### 1. Runtime Core

Introduce a new `src/runtime/` package built around these responsibilities:

- `SessionRuntime`
  Owns all per-`stream_id` runtime state and acts as the single source of truth for a chat session.
- `SessionState`
  Holds `ChatStream`, last message, scheduling/debounce state, reply pipeline instances, and runtime metadata.
- `RuntimeManager`
  Owns the in-memory registry of active sessions and routes incoming messages to the right session runtime.

This replaces the current split ownership in:

- `ChatManager.streams`
- `ChatManager.last_messages`
- `Heartflow.heartflow_chat_list`
- `ReplyerManager._repliers`

### 2. Direct Message Flow

Incoming messages should be handled in this order:

1. parse and normalize message
2. run pre-process events
3. resolve/create session state
4. persist message for audit/history
5. update session runtime state
6. execute command path if matched
7. run decision/reply pipeline directly from in-memory state
8. persist outgoing replies

The database remains the durable history store, but no longer acts as a queue between receive and decision stages.

### 3. Thin Extension System

Collapse the current plugin framework into a simpler extension layer:

- `CommandRegistry`
- `ActionRegistry`
- `ToolRegistry`
- `EventBus`

Built-ins register themselves into these registries directly. The system does not need:

- plugin class instantiation lifecycle
- manifest-driven runtime registration as the primary execution model
- `apis/*` wrappers for internal calls
- multiple registry/manager layers for the same component types

Manifest/config files can still be read where useful, but runtime wiring should become direct and explicit.

### 4. Reply Pipeline

Refactor reply generation into staged components:

- context collection
- memory retrieval
- tool execution
- prompt assembly
- model generation
- post-processing and sending

`BaseReplyer` becomes a facade over smaller pipeline parts instead of a monolith.

### 5. Memory Pipeline

Split `memory_retrieval.py` into focused modules:

- question generation
- retrieval execution
- cache/thinking-back reuse
- result formatting

Keep current database tables and vector store schema, but make runtime boundaries explicit. Memory writes should expose failure clearly instead of silently succeeding after partial async work.

### 6. Test Harness

Add a minimal but real test framework using:

- `pytest`
- `pytest-asyncio`

Initial test coverage targets:

- session state lifecycle
- event bus ordering/interception
- command dispatch
- runtime message flow with fake storage and fake sender

## Migration Strategy

Do the migration in one implementation track, but structure code so the new runtime becomes the active path early:

1. introduce new runtime package and test harness
2. introduce thin registries/event bus
3. route `ChatBot.message_process` into the new runtime
4. remove DB polling from the active decision loop
5. migrate built-ins and runtime dispatch from old plugin framework to the thin extension system
6. shrink old framework modules into compatibility shells or remove them when dead

## Non-Goals For This Pass

- preserving old internal plugin APIs
- preserving old extension loading semantics
- perfectly decomposing every large file in the same patch if they are no longer on the active path

## Acceptance Criteria

- one active in-memory runtime source per `stream_id`
- no active receive -> DB -> poll DB -> decide loop
- active runtime no longer depends on the old thick plugin managers
- tests exist and run for the new runtime boundary
- config files and database schema remain usable without migration
