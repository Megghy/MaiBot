# Frontend Optimization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce the dashboard's initial payload and simplify frontend state/auth flow without changing product behavior.

**Architecture:** Keep the current React + TanStack Router structure, but move route loading to lazy boundaries, centralize token/auth request helpers, and trim high-churn page logic where the current code is obviously repetitive. Prefer local refactors over introducing new state libraries or broad redesigns.

**Tech Stack:** React 19, TanStack Router, Vite, TypeScript, Tailwind, Bun

---

## Scope

1. Route-level lazy loading for page components.
2. Protected-route auth guard at router level instead of layout-only redirect.
3. Shared token/request helpers for pages still hand-rolling auth headers.
4. Targeted cleanup in high-complexity routes with immediate payoff.
5. Fresh build verification and bundle-size comparison.

## Task 1: Router and Chunk Boundaries

**Files:**
- Modify: `MaiBot-Dashboard/src/router.tsx`

**Steps:**
1. Replace eager route component imports with `lazyRouteComponent(...)`.
2. Add a small shared loading fallback around the router outlet.
3. Move auth enforcement to the protected route's `beforeLoad`.
4. Keep `/auth` and `/setup` reachable without the protected layout.

## Task 2: Shared Auth/Request Utilities

**Files:**
- Modify: `MaiBot-Dashboard/src/lib/fetch-with-auth.ts`
- Modify: `MaiBot-Dashboard/src/hooks/use-auth.ts`
- Modify: `MaiBot-Dashboard/src/components/layout.tsx`
- Modify: `MaiBot-Dashboard/src/routes/auth.tsx`
- Modify: `MaiBot-Dashboard/src/routes/setup/api.ts`

**Steps:**
1. Add shared helpers for reading/writing/clearing the access token.
2. Reuse those helpers in auth checks and logout flows.
3. Remove duplicate token access where a shared helper already covers it.

## Task 3: API Usage Cleanup

**Files:**
- Modify: `MaiBot-Dashboard/src/lib/plugin-api.ts`
- Modify: `MaiBot-Dashboard/src/routes/plugin-mirrors.tsx`
- Modify: `MaiBot-Dashboard/src/routes/index.tsx`
- Modify: `MaiBot-Dashboard/src/routes/settings.tsx`

**Steps:**
1. Move plugin mirror CRUD requests behind shared API helpers.
2. Replace dashboard page's hand-written auth request with shared fetch helpers.
3. Remove duplicated auth header assembly in settings-related requests where possible.
4. Avoid broad data-layer rewrites; only normalize the currently inconsistent hot paths.

## Task 4: High-Value Page Simplification

**Files:**
- Modify: `MaiBot-Dashboard/src/routes/config/bot.tsx`
- Modify: `MaiBot-Dashboard/src/routes/logs.tsx`
- Modify: `MaiBot-Dashboard/src/lib/log-websocket.ts`

**Steps:**
1. Collapse obviously repetitive bot-config autosave wiring into a smaller, shared pattern.
2. Reduce log viewer recomputation pressure for frequent updates/search filtering.
3. Keep behavior unchanged; no new backend contracts.

## Task 5: Verification

**Files:**
- Verify only

**Steps:**
1. Run frontend build from `MaiBot-Dashboard`.
2. Inspect generated chunk sizes and confirm the main entry chunk drops materially.
3. Run lint if the touched files remain compatible with the current repo rules.
4. Report residual risks if any heavy route chunk still needs a second-pass split.
