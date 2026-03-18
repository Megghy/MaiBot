# Lifecycle Controls Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add authenticated WebUI restart and shutdown controls backed by a unified launcher/worker lifecycle flow.

**Architecture:** Introduce a worker lifecycle controller inside the running process and move process restart responsibility to a launcher wrapper in `bot.py`. WebUI system routes will request lifecycle actions instead of directly replacing the process.

**Tech Stack:** Python 3, asyncio, FastAPI, existing WebUI frontend bundle source, Uvicorn

---

### Task 1: Add lifecycle state management

**Files:**
- Create: `src/runtime/lifecycle.py`

**Step 1: Define lifecycle request state**

- Add an enum-like representation for `none`, `shutdown`, `restart`.
- Add an `asyncio.Event` driven controller singleton.

**Step 2: Expose minimal API**

- Implement:
  - `request_shutdown()`
  - `request_restart()`
  - `wait_for_request()`
  - `consume_requested_action()`
  - `reset_lifecycle_state()`

**Step 3: Keep behavior fail-fast**

- Reject conflicting requests by keeping the first request and logging the later one.

### Task 2: Move process control to `bot.py`

**Files:**
- Modify: `bot.py`

**Step 1: Split current startup into worker execution**

- Wrap existing initialize/run/shutdown logic into a worker function that returns an exit code.

**Step 2: Add launcher loop**

- Launcher starts worker with `subprocess.Popen`.
- Wait for worker exit code.
- Restart only when exit code matches `RESTART_EXIT_CODE`.

**Step 3: Preserve console behavior**

- On launcher `KeyboardInterrupt`, forward termination to worker and wait briefly.

### Task 3: Connect worker runtime to lifecycle requests

**Files:**
- Modify: `src/main.py`
- Modify: `src/runtime/startup.py`
- Modify: `bot.py`

**Step 1: Stop using unbounded `while True` gather loop**

- Build service tasks once per worker run.
- Wait on both service completion and lifecycle request event.

**Step 2: Handle requested action**

- On `shutdown` or `restart`, cancel service tasks cleanly.
- Call `graceful_shutdown()`.
- Return the mapped exit code to launcher.

**Step 3: Keep abnormal failures visible**

- If a service task exits unexpectedly, log and propagate non-zero exit.

### Task 4: Secure and extend WebUI system routes

**Files:**
- Modify: `src/webui/routers/system.py`

**Step 1: Add shared token verification**

- Mirror existing WebUI route pattern with `Authorization: Bearer <token>`.

**Step 2: Replace direct `os.execv` restart**

- `POST /system/restart` should call runtime lifecycle request API and return an accepted response.

**Step 3: Add shutdown API**

- `POST /system/shutdown` should request shutdown and return an accepted response.

### Task 5: Add frontend controls

**Files:**
- Modify: frontend source file(s) in `MaiBot-Dashboard` or `webui` source that render system actions
- Rebuild production bundle if this repo stores built assets

**Step 1: Locate current system control UI**

- Reuse existing API wrapper and restart overlay behavior.

**Step 2: Add two explicit actions**

- `重启后端`
- `关闭后端`

**Step 3: Add request/feedback states**

- Confirmation before firing the request.
- Restart overlay for restart.
- Persistent “backend is shut down” state for shutdown.

### Task 6: Verify end-to-end behavior

**Files:**
- None

**Step 1: Static verification**

- Run config load / import level verification for changed Python modules.

**Step 2: Behavioral verification**

- Run a short launcher/worker smoke test.
- Verify:
  - invalid token is rejected
  - restart request produces restart exit code path
  - shutdown request produces normal exit path

**Step 3: Frontend verification**

- Build the frontend bundle if source changed.

