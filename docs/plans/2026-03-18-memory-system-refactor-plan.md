# Memory System Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the current script-like memory retrieval and person-memory flows with clear, reusable pipelines whose read/write directions are explicit.

**Architecture:** Keep the current runtime chain, but refactor the memory system into stable layers: `Retrieval Orchestrator`, `Retrieval Sources`, `Memory Ingester`, and `Person/Memory Repository`. The first phase introduces a unified retrieval source interface and rewires `memory_retrieval.py` to orchestrate sources instead of directly reaching into multiple modules.

**Tech Stack:** Python, asyncio, existing vector store + Peewee models, current LLM APIs, pytest, ruff

---

## Progress

Current status on 2026-03-18:
- Document created
- Phase 1 started
- Unified retrieval source interface landed for:
  - recent query history
  - cached memories
  - concept expansion
  - LPMM prefetch
- `src/memory_system/memory_retrieval.py` now consumes those paths through a shared orchestrator helper instead of direct ad-hoc calls

Remaining work for Phase 1:
- move more retrieval paths behind the same source contract
- decide whether agent tools should directly reuse the same source backends or remain adapters
- continue shrinking `memory_retrieval.py`

---

## Target Architecture

### 1. Retrieval Flow

```
NeedMemory
-> Build Retrieval Request
-> Fan out to Retrieval Sources
-> Merge / Deduplicate / Rank
-> Optional Agent Fallback
-> Prompt Block
```

Planned source categories:
- recent query history
- cached answers
- concept / jargon expansion
- chat history
- person profile
- person memory
- LPMM knowledge

### 2. Write / Learn Flow

```
Observation / QA Result
-> MemoryIngester
-> Write Request
-> Deduper
-> Repository
-> Vector Sync
```

This flow must be separate from retrieval. Retrieval should not directly mutate memory stores.

### 3. Person Layer

`Person` should remain a thin entity/facade. Heavy logic should live in repository/services modules, not in one giant memory ops file.

---

## Phase Plan

### Phase 1: Unified Retrieval Sources

**Goal**
- Standardize non-agent retrieval inputs/outputs.
- Turn `src/memory_system/memory_retrieval.py` into an orchestrator instead of a script-like module.

**Files**
- Create: `src/memory_system/retrieval_sources.py`
- Modify: `src/memory_system/memory_retrieval.py`
- Test: `tests/memory/test_retrieval_sources.py`

**Implementation**
- Define a source protocol / base shape:
  - request dataclass
  - result dataclass
  - async source interface
- Move these prefetch-style sources behind the interface:
  - recent query history
  - cached memories
  - concept retrieval
  - LPMM prefetch
- Add an orchestrator helper that executes a list of sources and returns normalized results.
- Rewire `memory_retrieval.py` to:
  - fetch recent query history via source
  - fetch concept expansion via source
  - fetch cached memories via source
  - fetch LPMM pre-query via source in `_process_single_question`

**Success Criteria**
- `memory_retrieval.py` no longer imports cache / concept / lpmm helpers as ad-hoc control flow everywhere.
- New sources return a common result shape.
- Existing behavior remains compatible.

### Phase 2: Split Retrieval from Learning

**Goal**
- Remove post-query writeback side effects from retrieval orchestration.

**Files**
- Modify: `src/memory_system/memory_retrieval.py`
- Modify: `src/memory_system/retrieval_analysis.py`
- Create: `src/memory_system/memory_ingester.py`
- Test: `tests/memory/test_memory_ingester.py`

**Implementation**
- Introduce `MemoryIngester` entrypoints for:
  - QA-derived person memory
  - QA-derived jargon
- Make retrieval return structured outputs only.
- Trigger ingestion explicitly from callers or a small async hook boundary, not from deep inside retrieval flow.

### Phase 3: Rework Person Memory Storage

**Goal**
- Break `memory_ops.py` into repository-like parts and unify global/group write flow.

**Files**
- Split from: `src/person_info/memory_ops.py`
- Create:
  - `src/person_info/memory_repository.py`
  - `src/person_info/memory_deduper.py`
  - `src/person_info/memory_vector_sync.py`
- Modify: `src/person_info/person_info.py`
- Test: `tests/person_info/test_memory_repository.py`

**Implementation**
- Extract shared write algorithm once.
- Model memory scope explicitly instead of duplicating global/group branches.
- Keep `Person` as a thin facade.

### Phase 4: Unify Agent Tool Sources

**Goal**
- Reuse the same retrieval source contracts for agent-facing tools.

**Files**
- Modify: `src/memory_system/retrieval_tools/*.py`
- Modify: `src/memory_system/retrieval_agent.py`
- Test: `tests/memory/test_retrieval_agent_sources.py`

**Implementation**
- Make agent tools thin adapters over retrieval source backends.
- Stop duplicating source-specific formatting logic in tools and orchestrator separately.

---

## Current Execution Decision

This session starts **Phase 1** immediately.

Phase 1 is the best first cut because:
- it improves flow clarity without changing too many persistence semantics
- it creates reusable contracts for later phases
- it reduces the ad-hoc nature of `memory_retrieval.py`
