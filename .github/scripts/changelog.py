#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Generate and extract CHANGELOG.md entries from Conventional Commit PR history."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess  # nosec B404
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_REPO = "open-edge-platform/anomalib"
CHANGELOG_SECTIONS = ("Added", "Removed", "Changed", "Deprecated", "Fixed")

# Strip leading emoji/symbol prefixes allowed in PR titles before conventional header.
_LEADING_EMOJI_RE = re.compile(
    r"^[\s\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]+",
)
_CONVENTIONAL_RE = re.compile(
    r"^(?P<type>[a-z]+)(?:\((?P<scope>[^)]+)\))?(?P<breaking>!)?:\s*(?P<desc>.+?)(?:\s+\(#(?P<pr>\d+)\))?$",
)
_BREAKING_FOOTER_RE = re.compile(r"^BREAKING[ -]CHANGE:\s*(.+)", re.MULTILINE)
_RELEASE_HOUSEKEEPING_RE = re.compile(
    r"^(?:prepare|prep|restore|sync|bump version)\b",
    re.IGNORECASE,
)
_VERSION_HEADING_RE = re.compile(r"^## \[(?P<label>[^\]]+)\](?:\s*-\s*(?P<date>.+))?$")


@dataclass(frozen=True)
class CommitEntry:
    """Parsed squash-merge commit metadata."""

    sha: str
    commit_type: str
    scope: str | None
    breaking: bool
    description: str
    pr_number: int | None
    git_author: str
    breaking_note: str | None = None


@dataclass(frozen=True)
class ChangelogItem:
    """Formatted changelog bullet assigned to a Keep a Changelog section."""

    section: str
    emoji: str
    scope: str
    summary: str
    author: str | None
    pr_number: int | None

    def format_bullet(self, repo: str) -> str:
        """Render the bullet line in Anomalib's CHANGELOG.md format."""
        parts = [f"- {self.emoji} **{self.scope}**: {self.summary}"]
        if self.author:
            parts.append(f"by @{self.author}")
        if self.pr_number is not None:
            parts.append(f"in https://github.com/{repo}/pull/{self.pr_number}")
        return " ".join(parts)


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


def _run_git(args: list[str], cwd: Path | None = None) -> str:
    """Run a git command safely without invoking a shell."""
    git_bin = shutil.which("git")
    if git_bin is None:
        msg = "git executable not found on PATH"
        raise RuntimeError(msg)
    result = subprocess.run(  # noqa: S603  # nosec B603
        [git_bin, *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def find_latest_lib_tag(exclude_version: str | None = None, cwd: Path | None = None) -> str:
    """Return the latest ``lib/v*.*.*`` tag, optionally excluding a version being prepared."""
    output = _run_git(["tag", "-l", "lib/v*.*.*", "--sort=-v:refname"], cwd=cwd)
    excluded = f"lib/v{normalize_version(exclude_version)}" if exclude_version else None
    for line in output.splitlines():
        tag = line.strip()
        if not tag:
            continue
        if excluded and tag == excluded:
            continue
        return tag
    msg = "No matching 'lib/v*.*.*' git tags found."
    raise RuntimeError(msg)


def get_git_commits(rev_range: str, cwd: Path | None = None) -> list[tuple[str, str, str, str]]:
    """Return ``(sha, subject, body, author_name)`` tuples for commits in ``rev_range``."""
    raw = _run_git(
        ["log", "--first-parent", "--format=%H%x1f%s%x1f%b%x1f%an%x1e", rev_range],
        cwd=cwd,
    )
    commits: list[tuple[str, str, str, str]] = []
    for raw_record in raw.split("\x1e"):
        cleaned_record = raw_record.strip("\n")
        if not cleaned_record:
            continue
        fields = cleaned_record.split("\x1f")
        if len(fields) != 4:
            continue
        sha, subject, body, author = fields
        commits.append((sha.strip(), subject.strip(), body.strip(), author.strip()))
    return commits


def parse_commit(sha: str, subject: str, body: str, git_author: str) -> CommitEntry | None:
    """Parse a Conventional Commit squash-merge header/body into a ``CommitEntry``."""
    clean_subject = _LEADING_EMOJI_RE.sub("", subject).strip()
    match = _CONVENTIONAL_RE.match(clean_subject)
    if match is None:
        return None

    commit_type = match.group("type")
    scope = match.group("scope")
    breaking = bool(match.group("breaking"))
    desc = match.group("desc").strip()
    pr_str = match.group("pr")
    pr_number = int(pr_str) if pr_str is not None else None

    breaking_match = _BREAKING_FOOTER_RE.search(body)
    breaking_note = breaking_match.group(1).strip() if breaking_match else None
    if breaking_note:
        breaking = True

    return CommitEntry(
        sha=sha,
        commit_type=commit_type,
        scope=scope,
        breaking=breaking,
        description=desc,
        pr_number=pr_number,
        git_author=git_author,
        breaking_note=breaking_note,
    )


def is_housekeeping_commit(entry: CommitEntry) -> bool:
    """Return True for release-prep commits that should not appear in release notes."""
    return (
        entry.commit_type == "chore"
        and entry.scope in {"release", "lib"}
        and bool(
            _RELEASE_HOUSEKEEPING_RE.match(entry.description),
        )
    )


def _format_summary(description: str) -> str:
    """Capitalize the first character of a description while keeping acronyms intact."""
    if not description:
        return description
    return description[0].upper() + description[1:] if description[0].islower() else description


_TYPE_MAP: dict[str, tuple[str, str]] = {
    "feat": ("Added", "🚀"),
    "perf": ("Changed", "⚡"),
    "refactor": ("Changed", "🔄"),
    "ci": ("Changed", "🔧"),
}


def classify_commit(entry: CommitEntry, *, include_internal: bool = False) -> tuple[str, str] | None:
    """Map a parsed commit to ``(Keep-a-Changelog section, emoji)``, or ``None`` to omit."""
    if is_housekeeping_commit(entry):
        return None

    desc_lower = entry.description.lower()
    is_security = entry.scope == "security" or "[security]" in desc_lower or "security fix" in desc_lower

    result: tuple[str, str] | None = None
    if entry.breaking:
        section = "Removed" if desc_lower.startswith(("remove", "drop")) else "Changed"
        result = (section, "⚠️")
    elif entry.commit_type == "fix":
        result = ("Fixed", "🔒" if is_security else "🐞")
    elif entry.commit_type == "docs":
        section = "Fixed" if desc_lower.startswith(("fix", "correct", "clarify", "restore")) else "Changed"
        result = (section, "📚")
    elif entry.commit_type in {"build", "chore"}:
        default_emoji = "📦" if entry.commit_type == "build" else "🔧"
        result = ("Changed", "🔒" if is_security else default_emoji)
    elif entry.commit_type in _TYPE_MAP:
        result = _TYPE_MAP[entry.commit_type]
    elif include_internal and entry.commit_type in {"test", "style"}:
        result = ("Changed", "🧪" if entry.commit_type == "test" else "🎨")

    return result


def _resolve_github_token() -> str | None:
    """Return a GitHub token from environment variables or ``gh auth token`` if available."""
    for env_var in ("GH_TOKEN", "GITHUB_TOKEN"):
        token = os.environ.get(env_var, "").strip()
        if token:
            return token

    gh_bin = shutil.which("gh")
    if gh_bin is not None:
        result = subprocess.run(  # noqa: S603  # nosec B603
            [gh_bin, "auth", "token"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    return None


def fetch_pr_author(repo: str, pr_number: int, token: str | None = None) -> str | None:
    """Look up the PR author login from GitHub's REST API."""
    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}"
    if not url.startswith("https://api.github.com/"):
        msg = f"Refusing unexpected URL: {url}"
        raise ValueError(msg)

    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "anomalib-changelog-generator",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = urllib.request.Request(url, headers=headers)  # noqa: S310
    try:
        with urllib.request.urlopen(request, timeout=15) as response:  # noqa: S310  # nosec B310
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"Warning: could not fetch PR #{pr_number} metadata ({exc}); falling back.", file=sys.stderr)
        return None

    user = payload.get("user") or {}
    login = user.get("login")
    if not isinstance(login, str) or not login:
        return None
    if login.startswith("app/"):
        login = f"{login[len('app/') :]}[bot]"
    return login


def build_release_section(
    version: str,
    release_date: str,
    entries: list[CommitEntry],
    *,
    repo: str = DEFAULT_REPO,
    author_lookup: dict[int, str] | None = None,
    use_github_api: bool = True,
    include_internal: bool = False,
) -> str:
    """Render a Keep a Changelog version section from parsed commit entries."""
    bare_version = normalize_version(version)
    grouped: dict[str, list[ChangelogItem]] = {section: [] for section in CHANGELOG_SECTIONS}
    cache: dict[int, str | None] = dict(author_lookup or {})
    token = _resolve_github_token() if use_github_api else None

    for entry in entries:
        classified = classify_commit(entry, include_internal=include_internal)
        if classified is None:
            continue
        section, emoji = classified

        author: str | None = None
        if entry.pr_number is not None:
            if entry.pr_number not in cache and use_github_api:
                cache[entry.pr_number] = fetch_pr_author(repo, entry.pr_number, token=token)
            author = cache.get(entry.pr_number)

        scope = entry.scope or ("breaking" if entry.breaking else entry.commit_type)
        summary = _format_summary(entry.description)
        if entry.breaking_note and entry.breaking_note.lower() not in summary.lower():
            summary = f"{summary} ({entry.breaking_note})"

        grouped[section].append(
            ChangelogItem(
                section=section,
                emoji=emoji,
                scope=scope,
                summary=summary,
                author=author,
                pr_number=entry.pr_number,
            ),
        )

    lines: list[str] = [f"## [v{bare_version}] - {release_date}"]
    for section in CHANGELOG_SECTIONS:
        items = grouped[section]
        if not items:
            continue
        lines.extend(["", f"### {section}", ""])
        lines.extend(item.format_bullet(repo) for item in items)

    return "\n".join(lines).rstrip() + "\n"


def update_changelog_content(existing: str, version: str, new_section: str) -> str:
    """Insert or replace ``## [v<version>]`` in ``CHANGELOG.md`` content."""
    bare_version = normalize_version(version)
    lines = existing.splitlines()
    new_section_lines = new_section.strip("\n").splitlines()

    # Replace existing matching version block if re-run idempotently.
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

    unreleased_note = "<!-- Release notes are auto-generated from Conventional Commit PR titles at release time. -->"

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
            prefix = [*lines[: unreleased_idx + 1], "", unreleased_note]
        else:
            prefix = list(lines[:first_release_idx])
            while prefix and not prefix[-1].strip():
                prefix.pop()
        updated_lines = [*prefix, "", *new_section_lines, "", *lines[first_release_idx:]]
    elif unreleased_idx is not None:
        prefix = [*lines[: unreleased_idx + 1], "", unreleased_note]
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
            unreleased_note,
        ]
        updated_lines = [*header, "", *new_section_lines]

    return "\n".join(updated_lines).rstrip() + "\n"


def _strip_empty_subsections(section_lines: list[str]) -> list[str]:
    """Remove empty ``### <Subsection>`` blocks when extracting legacy changelog entries."""
    cleaned: list[str] = []
    idx = 0
    while idx < len(section_lines):
        line = section_lines[idx]
        if line.startswith("### "):
            block_end = idx + 1
            while block_end < len(section_lines) and not section_lines[block_end].startswith("### "):
                block_end += 1
            body = [item for item in section_lines[idx + 1 : block_end] if item.strip()]
            if body:
                if cleaned and cleaned[-1].strip():
                    cleaned.append("")
                cleaned.append(line)
                cleaned.append("")
                cleaned.extend(body)
            idx = block_end
        else:
            if line.strip() or (cleaned and cleaned[-1].strip()):
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


def bump_pyproject_version(pyproject_content: str, version: str) -> str:
    """Update ``version = "..."`` inside ``[project]`` of ``pyproject.toml``."""
    bare_version = normalize_version(version)
    lines = pyproject_content.splitlines()
    in_project = False
    replaced = False

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_project = stripped == "[project]"
            continue
        if in_project and re.match(r'^version\s*=\s*"[^"]+"', stripped):
            lines[idx] = f'version = "{bare_version}"'
            replaced = True
            break

    if not replaced:
        msg = "Could not find `[project]` `version = ...` field in pyproject.toml."
        raise ValueError(msg)

    return "\n".join(lines).rstrip() + "\n"


def _cmd_generate(args: argparse.Namespace) -> int:
    """Handle the ``generate`` subcommand."""
    cwd = Path.cwd()
    bare_version = normalize_version(args.version)
    since_tag = args.since_tag or find_latest_lib_tag(exclude_version=bare_version, cwd=cwd)
    rev_range = f"{since_tag}..{args.to_ref}"

    raw_commits = get_git_commits(rev_range, cwd=cwd)
    entries = [
        parsed
        for sha, subject, body, git_author in raw_commits
        if (parsed := parse_commit(sha, subject, body, git_author)) is not None
    ]

    release_date = args.date or datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    section = build_release_section(
        version=bare_version,
        release_date=release_date,
        entries=entries,
        repo=args.repo,
        use_github_api=not args.no_github_api,
        include_internal=args.include_internal,
    )

    if args.update:
        changelog_path = Path(args.changelog)
        existing = changelog_path.read_text(encoding="utf-8") if changelog_path.exists() else ""
        updated = update_changelog_content(existing, bare_version, section)
        changelog_path.write_text(updated, encoding="utf-8")
        print(f"Updated {changelog_path} with [v{bare_version}] from {rev_range} ({len(entries)} commits).")

        if args.pyproject:
            pyproject_path = Path(args.pyproject)
            pyproject_updated = bump_pyproject_version(
                pyproject_path.read_text(encoding="utf-8"),
                bare_version,
            )
            pyproject_path.write_text(pyproject_updated, encoding="utf-8")
            print(f"Updated {pyproject_path} version to {bare_version}.")
    elif args.output and args.output != "-":
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(section, encoding="utf-8")
    else:
        sys.stdout.write(section)

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
        description="Generate and extract Anomalib CHANGELOG.md entries.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen_parser = subparsers.add_parser("generate", help="Generate a release section from git history.")
    gen_parser.add_argument("--version", required=True, help="Target version (e.g. 2.7.2 or v2.7.2).")
    gen_parser.add_argument("--since-tag", help="Starting git tag (defaults to latest lib/v*.*.* tag).")
    gen_parser.add_argument("--to-ref", default="HEAD", help="Ending git ref (default: HEAD).")
    gen_parser.add_argument("--date", help="Release date YYYY-MM-DD (default: today UTC).")
    gen_parser.add_argument("--repo", default=DEFAULT_REPO, help="GitHub repository owner/name.")
    gen_parser.add_argument("--changelog", default="CHANGELOG.md", help="Path to CHANGELOG.md.")
    gen_parser.add_argument("--update", action="store_true", help="Write the generated section into CHANGELOG.md.")
    gen_parser.add_argument(
        "--pyproject",
        help="Optional pyproject.toml path to also bump [project].version when --update is set.",
    )
    gen_parser.add_argument("--output", default="-", help="Output path when not using --update (default: stdout).")
    gen_parser.add_argument("--no-github-api", action="store_true", help="Skip GitHub API PR author lookup.")
    gen_parser.add_argument(
        "--include-internal",
        action="store_true",
        help="Include test/style commits in addition to user-facing changes.",
    )
    gen_parser.set_defaults(func=_cmd_generate)

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
