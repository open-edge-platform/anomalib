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
- the PR title does not accurately summarize the user-facing change (since `CHANGELOG.md` and release notes are auto-generated from Conventional Commit PR titles at release time);
- a regular PR edits `CHANGELOG.md` unnecessarily (unless it is a release-preparation PR);
- a release-preparation PR's `CHANGELOG.md` entry is missing its change-type emoji, bold scope, summary, author attribution, or PR URL;
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

- Regular PRs should **not** edit `CHANGELOG.md`. Instead, `CHANGELOG.md` and GitHub Release notes are auto-generated from squash-merged Conventional Commit PR titles at release time via `git-cliff` (`cliff.toml`) and `.github/scripts/changelog.py` (and mirrored into GitHub Releases by `.github/workflows/release-lib.yaml`).
- Reviewers should check that the **PR title** accurately describes the user-facing change and that breaking changes are marked with `!` or a `BREAKING CHANGE:` footer.
- When reviewing a release-preparation PR (`docs(changelog): prepare CHANGELOG.md for v...` / `chore(release): ...`):
  - Use the existing Keep a Changelog section headings: `Added`, `Removed`, `Changed`, `Deprecated`, `Fixed`.
  - Verify each entry has a change-type emoji (for example, `🐞` for a bug fix), bold scope, user-facing summary, author attribution, and the actual PR URL:

    ```markdown
    - 🐞 **data**: Describe the user-facing fix by @github-user in https://github.com/open-edge-platform/anomalib/pull/1234
    ```

  - Verify attribution and PR references against PR metadata; do not invent them. Grouped entries may credit multiple authors and PRs.
  - Keep neighboring bullets contiguous without an extra blank line. Review added or modified entries without demanding unrelated historical rewrites.

## Repo-grounded review anchors

- `CONTRIBUTING.md`
- `docs/source/markdown/guides/developer/contributing.md`
- `.agents/skills/python-docstrings/SKILL.md`
- `CHANGELOG.md`

## Review prompts

- Does the code change require docstring, README, docs page, or example updates?
- Are docstrings informative enough for users to understand behavior and expected inputs?
- Does the PR title accurately describe the user-facing change for release-note generation?
- If a public symbol or module changed, is the reference documentation still accurate?

## Reviewer checklist

- Check docstrings for public APIs.
- Check README, docs, and examples for user-facing changes.
- Check PR title clarity (and `CHANGELOG.md` only on release-preparation PRs).
- Check that docs match the current API and workflow.
