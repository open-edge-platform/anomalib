---
name: docs-changelog
description: Reviews anomalib docstrings, documentation updates, and changelog expectations
---

# Anomalib Documentation and Changelog Review

Use this skill when reviewing docstrings, user docs, examples, READMEs, and release-note impact.

## Purpose and scope

Use this skill when code changes may affect user-visible documentation, examples, READMEs, or release notes.

## Request changes when

- user-facing behavior changes without matching docs updates;
- public APIs change without docstring or reference-doc updates;
- a significant user-facing change is missing a `CHANGELOG.md` entry under `## [Unreleased]`;
- a new or modified changelog entry is missing its change-type emoji, bold scope, summary, author attribution, or PR URL;
- examples or README usage snippets no longer match the actual API.

## Docstrings

- Public Python APIs should use Google-style docstrings.
- Use the existing `python-docstrings` skill for docstring formatting details.
- Ask for docstrings when behavior is non-trivial, user-facing, or part of a reusable API surface.
- For tensors, arrays, batches, or structured outputs, ask reviewers to document shapes or field expectations when they matter for correct usage.

## Documentation updates

- If a PR changes APIs, CLI behavior, model behavior, config structure, workflows, or outputs, ask for related documentation updates.
- Review the nearest documentation surface, not just the edited Python file: `README.md`, docs under `docs/source/markdown/`, model-specific `README.md`, examples, or reference pages.
- Prefer small, precise doc updates over broad rewrites.

## Changelog

- Significant user-facing changes should update `CHANGELOG.md` under `## [Unreleased]`.
- Use the existing Keep a Changelog section headings already present in the repo: `Added`, `Removed`, `Changed`, `Deprecated`, `Fixed`.
- Release-preparation PRs may move entries into a versioned section; regular PRs use `Unreleased`.
- Check the entry format, not only its presence. Changelog format is a release requirement, not a style nit to skip because linters run.
- Require a change-type emoji (for example, `🐞` for a bug fix), bold scope, user-facing summary, author attribution, and the actual PR URL:

  ```markdown
  - 🐞 **data**: Describe the user-facing fix by @github-user in https://github.com/open-edge-platform/anomalib/pull/1234
  ```

- Verify attribution and PR references against PR metadata; do not invent them. Grouped entries may credit multiple authors and PRs.
- The example uses placeholders; replace `@github-user` and PR number `1234` with the actual metadata.
- Keep neighboring bullets contiguous without an extra blank line. Review added or modified entries without demanding unrelated historical rewrites.
- Flag missing fields with a corrected example even when the implementation and tests are correct. PR title emoji rules are separate from changelog entry rules.
- Purely internal changes may not need a changelog entry, but reviewers should call out missing entries for behavior, API, docs, or user workflow changes.

## Repo-grounded review anchors

- `CONTRIBUTING.md`
- `docs/source/markdown/guides/developer/contributing.md`
- `.agents/skills/python-docstrings/SKILL.md`
- `CHANGELOG.md`

## Review prompts

- Does the code change require docstring, README, docs page, or example updates?
- Are docstrings informative enough for users to understand behavior and expected inputs?
- Should this change be recorded under `## [Unreleased]`?
- If a public symbol or module changed, is the reference documentation still accurate?

## Reviewer checklist

- Check docstrings for public APIs.
- Check README, docs, and examples for user-facing changes.
- Check `CHANGELOG.md` for significant changes.
- Check new or modified entries for the correct heading, emoji, bold scope, summary, author attribution, and PR URL.
- Check that docs match the current API and workflow.
