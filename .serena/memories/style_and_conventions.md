# Style and conventions
- User communication language is Chinese; code symbols are English.
- Runtime is Python despite AGENTS mentioning Bun for JS tooling in other contexts.
- Formatting/linting is configured with Ruff. `pyproject.toml` sets line-length 120, selects rules E/F/B, ignores E711/E501, and uses double quotes in formatter settings.
- Code style in repo heavily uses global managers/singletons, module-level instances, async/await, and direct logger usage via `src.common.logger.get_logger`.
- Many modules are monolithic (>500 lines) and use pragmatic direct coding rather than strict layering.
- Error handling often logs and continues instead of fail-fast in some persistence/query boundaries.