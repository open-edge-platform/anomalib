---
name: anomalib-pr-workflow
description: Reviews anomalib contributor workflow, PR title, branch naming, and quality gate expectations
---

# Anomalib PR Workflow Review

Use this skill when reviewing whether a change is ready for contribution and merge.

## Purpose and scope

Use this skill for merge readiness, contributor workflow, CI expectations, PR title checks, and branch naming.

## Request changes when

- the PR title does not follow the repository's Conventional Commit style;
- branch naming does not match `<type>/<scope>/<description>`;
- required quality gates, docs, tests, or changelog updates are missing;
- workflow or security-sensitive changes are not reviewed to the repo's standards.

## Quality gates

- Contributors are expected to run `prek run --all-files` and `pytest tests/` before finalizing a PR.
- Review feedback should align with the project's configured checks in `pyproject.toml` and `.pre-commit-config.yaml`.
- Security-sensitive changes should be reviewed with the repo's CI security tooling in mind: Bandit, CodeQL, Semgrep, Zizmor, Trivy, and Dependabot.

## PR titles and branch names

- PR titles should follow Conventional Commits as described in `CONTRIBUTING.md`.
- Branch names should follow `<type>/<scope>/<description>`.
- Allowed types and scopes should match the repository's Commitizen configuration in `pyproject.toml`.

## Reviewer expectations

- Ask for precise, actionable fixes grounded in repo policy rather than generic preferences.
- Escalate missing tests, missing docs, missing changelog entries, or workflow/security risks before approval.
- Be stricter when a PR changes CLI entrypoints, workflows, deployment, inferencers, or user-facing public APIs.

## Repo-grounded review anchors

- `CONTRIBUTING.md`
- `docs/source/markdown/guides/developer/contributing.md`
- `docs/source/markdown/guides/developer/code_review_checklist.md`
- `.pre-commit-config.yaml`
- `pyproject.toml`
- `SECURITY.md`

## Review prompts

- Is the PR title valid for the eventual squash commit?
- Are branch naming, changelog, tests, and docs in good shape for merge?
- Do the requested changes line up with the project's existing CI and security gates?

## Reviewer checklist

- Check PR title.
- Check branch naming.
- Check CI, test, doc, and changelog readiness.
- Check workflow and security-sensitive files more strictly.
