#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Validate PR titles against Anomalib's Conventional Commit format."""

from __future__ import annotations

import argparse
import os
import re
import sys

ALLOWED_TYPES = (
    "feat",
    "fix",
    "docs",
    "style",
    "refactor",
    "perf",
    "test",
    "build",
    "ci",
    "chore",
    "revert",
)

_LEADING_EMOJI_RE = re.compile(
    r"^[\s\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]+",
)
_PR_TITLE_RE = re.compile(
    r"^(?P<type>[a-z]+)(?:\((?P<scope>[a-zA-Z0-9_.\-/]+)\))?(?P<breaking>!)?:\s+(?P<desc>.+)$",
)


def strip_leading_emojis(title: str) -> str:
    """Remove optional leading emoji prefixes from a PR title."""
    return _LEADING_EMOJI_RE.sub("", title).strip()


def validate_pr_title(title: str) -> tuple[bool, str]:
    """Return ``(is_valid, error_or_clean_title)`` for a PR title."""
    if not title or not title.strip():
        return False, "PR title is empty."

    clean_title = strip_leading_emojis(title)
    match = _PR_TITLE_RE.match(clean_title)
    if match is None:
        return (
            False,
            f"Title {clean_title!r} must match '[optional emoji] <type>(<scope>): <description>'.",
        )

    commit_type = match.group("type")
    if commit_type not in ALLOWED_TYPES:
        allowed = ", ".join(ALLOWED_TYPES)
        return False, f"Unknown commit type {commit_type!r}. Allowed types: {allowed}."

    desc = match.group("desc").strip()
    if not desc:
        return False, "PR title description cannot be empty."

    return True, clean_title


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Validate a PR title against Conventional Commits.")
    parser.add_argument(
        "title",
        nargs="?",
        help="PR title to validate (defaults to PR_TITLE environment variable).",
    )
    args = parser.parse_args(argv)

    title = args.title if args.title is not None else os.environ.get("PR_TITLE", "")
    is_valid, detail = validate_pr_title(title)
    if not is_valid:
        print(f"::error::{detail}", file=sys.stderr)
        print(
            "::error::Format: [optional emoji] <type>(<scope>): <description>\n"
            "::error::Examples:\n"
            "::error::  🚀 feat(data): add support for custom dataset formats\n"
            "::error::  🐞 fix(model): resolve memory leak in PatchCore training\n"
            "::error::  📚 docs: update installation instructions",
            file=sys.stderr,
        )
        return 1

    print(f"PR title valid: {detail!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
