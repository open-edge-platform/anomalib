#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Insert git-cliff sections into CHANGELOG.md and extract release notes."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_VERSION_HEADING_RE = re.compile(r"^## \[(?P<label>[^\]]+)\](?:\s*-\s*(?P<date>.+))?$")
_UNRELEASED_NOTE = "<!-- Release notes are auto-generated from Conventional Commit PR titles at release time. -->"


def normalize_version(version: str) -> str:
    """Normalize a version or tag string like ``lib/v2.7.1`` or ``v2.7.1`` to ``2.7.1``."""
    cleaned = version.strip()
    if cleaned.startswith("lib/"):
        cleaned = cleaned[len("lib/") :]
    if cleaned.startswith("v"):
        cleaned = cleaned[1:]
    if not cleaned:
        msg = f"Invalid empty version string: {version!r}"
        raise ValueError(msg)
    return cleaned


def update_changelog_content(existing: str, version: str, new_section: str) -> str:
    """Insert or replace ``## [v<version>]`` in ``CHANGELOG.md`` content."""
    bare_version = normalize_version(version)
    lines = existing.splitlines()
    new_section_lines = new_section.strip("\n").splitlines()

    target_indices: tuple[int, int] | None = None
    first_release_idx: int | None = None
    unreleased_idx: int | None = None

    for idx, line in enumerate(lines):
        match = _VERSION_HEADING_RE.match(line)
        if match is None:
            continue
        label = match.group("label").strip()
        if label.lower() == "unreleased":
            unreleased_idx = idx
            continue
        if first_release_idx is None:
            first_release_idx = idx
        if normalize_version(label) == bare_version and target_indices is None:
            end_idx = len(lines)
            for next_idx in range(idx + 1, len(lines)):
                if _VERSION_HEADING_RE.match(lines[next_idx]):
                    end_idx = next_idx
                    break
            target_indices = (idx, end_idx)

    if target_indices is not None:
        start_idx, end_idx = target_indices
        tail = lines[end_idx:]
        while tail and not tail[0].strip():
            tail.pop(0)
        if tail:
            updated_lines = [*lines[:start_idx], *new_section_lines, "", *tail]
        else:
            updated_lines = [*lines[:start_idx], *new_section_lines]
    elif first_release_idx is not None:
        if unreleased_idx is not None:
            prefix = [*lines[: unreleased_idx + 1], "", _UNRELEASED_NOTE]
        else:
            prefix = list(lines[:first_release_idx])
            while prefix and not prefix[-1].strip():
                prefix.pop()
        updated_lines = [*prefix, "", *new_section_lines, "", *lines[first_release_idx:]]
    elif unreleased_idx is not None:
        prefix = [*lines[: unreleased_idx + 1], "", _UNRELEASED_NOTE]
        updated_lines = [*prefix, "", *new_section_lines]
    else:
        header = [
            "# Changelog",
            "",
            "All notable changes to this project will be documented in this file.",
            "",
            "The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).",
            "",
            "## [Unreleased]",
            "",
            _UNRELEASED_NOTE,
        ]
        updated_lines = [*header, "", *new_section_lines]

    return "\n".join(updated_lines).rstrip() + "\n"


def _strip_empty_subsections(section_lines: list[str]) -> list[str]:
    """Remove empty ``### <Subsection>`` blocks while preserving nonempty subsection bodies."""
    cleaned: list[str] = []
    idx = 0
    while idx < len(section_lines):
        line = section_lines[idx]
        if line.startswith("### "):
            block_end = idx + 1
            while block_end < len(section_lines) and not section_lines[block_end].startswith("### "):
                block_end += 1
            body = list(section_lines[idx + 1 : block_end])
            while body and not body[0].strip():
                body.pop(0)
            while body and not body[-1].strip():
                body.pop()
            if body:
                if cleaned and cleaned[-1].strip():
                    cleaned.append("")
                cleaned.append(line)
                cleaned.append("")
                cleaned.extend(body)
            idx = block_end
        else:
            cleaned.append(line)
            idx += 1
    return cleaned


def extract_release_notes(changelog_content: str, version: str) -> str:
    """Extract the exact markdown body for ``version`` from ``CHANGELOG.md`` content."""
    bare_version = normalize_version(version)
    lines = changelog_content.splitlines()

    start_idx: int | None = None
    end_idx = len(lines)

    for idx, line in enumerate(lines):
        match = _VERSION_HEADING_RE.match(line)
        if match is None:
            continue
        label = match.group("label").strip()
        if label.lower() == "unreleased":
            continue
        if start_idx is None:
            if normalize_version(label) == bare_version:
                start_idx = idx + 1
        else:
            end_idx = idx
            break

    if start_idx is None:
        msg = f"Version section 'v{bare_version}' not found in CHANGELOG.md."
        raise ValueError(msg)

    section_lines = _strip_empty_subsections(lines[start_idx:end_idx])
    while section_lines and not section_lines[0].strip():
        section_lines.pop(0)
    while section_lines and not section_lines[-1].strip():
        section_lines.pop()

    if not any(line.lstrip().startswith("- ") for line in section_lines):
        msg = f"Version section 'v{bare_version}' in CHANGELOG.md has no changelog entries."
        raise ValueError(msg)

    return "\n".join(section_lines) + "\n"


def _cmd_update(args: argparse.Namespace) -> int:
    """Handle the ``update`` subcommand."""
    bare_version = normalize_version(args.version)
    section = sys.stdin.read() if args.input == "-" else Path(args.input).read_text(encoding="utf-8")
    if not section.strip():
        print("Error: generated changelog section is empty.", file=sys.stderr)
        return 1

    changelog_path = Path(args.changelog)
    existing = changelog_path.read_text(encoding="utf-8") if changelog_path.exists() else ""
    updated = update_changelog_content(existing, bare_version, section)
    changelog_path.write_text(updated, encoding="utf-8")
    print(f"Updated {changelog_path} with [v{bare_version}].")
    return 0


def _cmd_extract(args: argparse.Namespace) -> int:
    """Handle the ``extract`` subcommand."""
    changelog_path = Path(args.changelog)
    if not changelog_path.exists():
        print(f"Error: changelog file not found: {changelog_path}", file=sys.stderr)
        return 1

    content = changelog_path.read_text(encoding="utf-8")
    try:
        notes = extract_release_notes(content, args.version)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.output and args.output != "-":
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(notes, encoding="utf-8")
    else:
        sys.stdout.write(notes)

    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Update and extract Anomalib CHANGELOG.md release sections.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    upd_parser = subparsers.add_parser("update", help="Insert or replace a version section in CHANGELOG.md.")
    upd_parser.add_argument("--version", required=True, help="Target version (e.g. 2.7.2 or v2.7.2).")
    upd_parser.add_argument("--input", default="-", help="Generated section file path (default: stdin).")
    upd_parser.add_argument("--changelog", default="CHANGELOG.md", help="Path to CHANGELOG.md.")
    upd_parser.set_defaults(func=_cmd_update)

    ext_parser = subparsers.add_parser("extract", help="Extract release notes for a version from CHANGELOG.md.")
    ext_parser.add_argument("--version", required=True, help="Version or tag (e.g. 2.7.1, v2.7.1, lib/v2.7.1).")
    ext_parser.add_argument("--changelog", default="CHANGELOG.md", help="Path to CHANGELOG.md.")
    ext_parser.add_argument("--output", default="-", help="Output path (default: stdout).")
    ext_parser.set_defaults(func=_cmd_extract)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
