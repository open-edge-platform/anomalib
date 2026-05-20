---
name: testing
description: Review/generate unit, integration, and regression test expectations
---

# Anomalib Testing Review/Generation

Use this skill when reviewing/generating additions or changes that should be covered by tests.

## Purpose and scope

Use this skill to decide what test coverage is required and whether existing tests still prove the intended behavior.

## Request changes when

- new behavior ships without tests;
- behavior changes update code but not existing tests;
- a bug fix has no regression test when one is feasible;
- tests are flaky, network-dependent, or placed in the wrong area of the suite.

## Test placement and scope

- Tests should live under `tests/` and generally mirror the code area they validate.
- Prefer the established split between `tests/unit/` and `tests/integration/`.
- Ask for unit tests for new behavior and for integration coverage when cross-component behavior changes.

## Test style

- Follow pytest conventions already used in the repo.
- Prefer fixtures and parametrization where they improve clarity and coverage.
- Keep tests offline and deterministic where practical.
- For tensor-heavy logic, ensure tests assert the properties that matter: shapes, values, reconstruction behavior, errors, or invariants.

## Regression mindset

- Bug fixes should include a regression test when feasible.
- Behavior changes should update existing tests, not only add new ones.
- If the change touches CLI/config/model loading or pipeline orchestration, review whether a higher-level test is also needed.

## Repo-grounded review anchors

- `pyproject.toml` for pytest markers and test path configuration

## Review prompts

- What test proves the new behavior works?
- What existing behavior could regress because of this change?
- Is the test placed in the right part of the suite?
- Does the change need both unit and integration coverage?

## Reviewer checklist

- Check that new behavior has test coverage.
- Check that changed behavior updates old tests too.
- Check placement under `tests/unit/` or `tests/integration/`.
- Check determinism and offline execution.
