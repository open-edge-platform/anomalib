---
name: python-style
description: Reviews anomalib Python style, typing, imports, and public API conventions
---

# Anomalib Python Style Review

Use this skill when reviewing Python code in `src/anomalib/` or `tests/`.

## Purpose and scope

This skill covers Python style, typing, imports, exports, copyright headers, and basic code hygiene.

## Core rules

- Target the repository's Python baseline.
- Follow the Ruff-configured line length of 120 characters.
- Match nearby anomalib code before suggesting stylistic rewrites.
- Prefer explicit, readable code over clever shortcuts.

## Request changes when

- public APIs are missing type annotations;
- new code introduces weak typing such as unnecessary `Any` or untyped public `**kwargs`;
- imports or exports drift from nearby package patterns;
- a touched Python file is missing the expected copyright/SPDX header;
- error handling becomes less explicit or debug code is left behind.

## Typing

- Public functions, methods, and constructors should have explicit type annotations.
- Prefer repository-established typing patterns such as `X | None`, `type[...]`, `Sequence[...]`, `TypeVar`, and `Generic` where they fit.
- Do not weaken types without a strong reason.
- Flag vague escape hatches such as unnecessary `Any`, broad untyped `**kwargs`, or type suppressions that hide real issues.

## Imports and exports

- Keep imports grouped as standard library, third-party, then local imports.
- Prefer absolute imports inside `anomalib`.
- When a public symbol is added to an `__init__.py`, verify that `__all__` stays accurate.

## Copyright and license header

- Python source files should include the standard Intel copyright and SPDX header used across the repository.
- For a **new** file, use the current year only, for example:
  - `# Copyright (C) 2026 Intel Corporation`
  - `# SPDX-License-Identifier: Apache-2.0`
- For an **existing** file updated in 2026, ensure the year or year range includes 2026.
  - Example: update `2024` to `2024-2026`.
  - Example: keep `2026` for a single-year file created in 2026.

## Error handling and code hygiene

- Catch specific exceptions instead of broad or silent failure patterns.
- Ask for explicit exceptions and informative error messages.
- Flag debug prints, dead code, commented-out code, and magic values that should be named constants or config.
- Prefer explicit validation with raised exceptions over fragile assumptions.

## Repo-grounded review anchors

- `pyproject.toml` defines Ruff, pydocstyle, mypy, pytest, and Commitizen expectations.

## Reviewer checklist

- Check typing on public APIs.
- Check imports and exports.
- Check the copyright/SPDX header on touched Python files.
- Check for obvious code hygiene regressions.
