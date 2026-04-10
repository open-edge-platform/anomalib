# Anomalib Copilot Instructions

Use these instructions when reviewing code changes in `anomalib`.

## Primary rule

Prefer the repository-local skills under `.agents/skills/` instead of embedding all review policy here.

## Purpose and scope

Use this file for repository-wide review guidance. Use the skill files below for topic-specific rules.

## Review priorities

- Keep reviews aligned with existing anomalib patterns instead of introducing new conventions.
- Prefer small, targeted feedback over broad refactors unless the PR explicitly aims to refactor.
- Treat `src/anomalib/`, `application/`, `tests/`, `docs/`, `.github/`, and relevant `.agents/skills/` files as first-class review surfaces.
- If a change touches `application/`, note that Anomalib Studio has some separate tooling and config from the root project.

## Minimum checklist

When writing review comments, explicitly check:

1. Correctness and edge cases
2. Architecture fit with anomalib patterns
3. Typing and public API clarity
4. Docstrings, docs, and changelog updates
5. Unit/integration coverage
6. Security and workflow risk
7. Maintainability and code hygiene

## Review tone

- Be specific and actionable.
- Prefer comments like "Please add/adjust X because anomalib expects Y" over generic style remarks.
- Ground feedback in the repository skills and the source documents they reference.

## Skills to use

- Python style, typing, imports, and API hygiene:
  - `.agents/skills/python-style/SKILL.md`
- Models, data, dataclasses, callbacks, metrics, and CLI integration:
  - `.agents/skills/models-data/SKILL.md`
- Docstrings, docs updates, and changelog expectations:
  - `.agents/skills/docs-changelog/SKILL.md`
  - `.agents/skills/python-docstrings/SKILL.md`
  - `.agents/skills/model-doc-sync/SKILL.md`
- Unit/integration/regression test expectations:
  - `.agents/skills/testing/SKILL.md`
- PR title, branch naming, contributor workflow, and quality gates:
  - `.agents/skills/pr-workflow/SKILL.md`
- Third-party code attribution and licensing:
  - `.agents/skills/third-party-code/SKILL.md`
- Benchmark and docs refresh:
  - `.agents/skills/benchmark-and-docs-refresh/SKILL.md`
- Model README and docs page sample image:
  - `.agents/skills/model-sample-image-export/SKILL.md`
