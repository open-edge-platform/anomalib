# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Unit and round-trip tests for `.github/scripts/changelog.py`."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / ".github" / "scripts" / "changelog.py"


def _load_changelog_module() -> ModuleType:
    """Load `.github/scripts/changelog.py` as a module."""
    spec = importlib.util.spec_from_file_location("anomalib_ci_changelog", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


changelog = _load_changelog_module()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2.7.1", "2.7.1"),
        ("v2.7.1", "2.7.1"),
        ("lib/v2.7.1", "2.7.1"),
        (" lib/v2.8.0 ", "2.8.0"),
    ],
)
def test_valid_versions(raw: str, expected: str) -> None:
    """Accepts bare, v-prefixed, and lib/v-prefixed versions."""
    assert changelog.normalize_version(raw) == expected


def test_empty_version_raises() -> None:
    """Rejects empty or prefix-only version strings."""
    with pytest.raises(ValueError, match="Invalid empty version"):
        changelog.normalize_version("lib/v")


def test_parse_emoji_prefixed_header() -> None:
    """Parses emoji-prefixed PR titles and extracts PR number."""
    entry = changelog.parse_commit(
        sha="abc123",
        subject="🐞 fix(data): accept valid split ratios (#3945)",
        body="",
        git_author="Contributor",
    )
    assert entry is not None
    assert entry.commit_type == "fix"
    assert entry.scope == "data"
    assert entry.breaking is False
    assert entry.description == "accept valid split ratios"
    assert entry.pr_number == 3945


def test_parse_breaking_footer() -> None:
    """Detects breaking changes from commit body footers."""
    entry = changelog.parse_commit(
        sha="def456",
        subject="chore(model)!: remove legacy exporter (#3826)",
        body="Some context\n\nBREAKING CHANGE: dynamo=False is no longer supported.",
        git_author="Maintainer",
    )
    assert entry is not None
    assert entry.breaking is True
    assert entry.breaking_note == "dynamo=False is no longer supported."
    assert changelog.classify_commit(entry) == ("Removed", "⚠️")


def test_housekeeping_commits_omitted() -> None:
    """Release housekeeping commits are omitted from generated notes."""
    for subject in (
        "chore(release): prepare 2.7.1 (#3930)",
        "chore(release): restore 2.7.2.dev0 (#3934)",
        "chore(release): sync main into feature/v2.9.0 (#3937)",
        "chore(lib): prep v2.7.0 release (#3823)",
    ):
        entry = changelog.parse_commit("000", subject, "", "anomalibbot")
        assert entry is not None
        assert changelog.is_housekeeping_commit(entry)
        assert changelog.classify_commit(entry) is None


def test_security_and_docs_classification() -> None:
    """Security updates use lock emoji and docs fixes map to Fixed."""
    sec = changelog.parse_commit("1", "chore(deps): update pillow to v13 [security] (#3939)", "", "bot")
    doc_fix = changelog.parse_commit("2", "docs(cli): clarify benchmark availability (#3931)", "", "bot")
    doc_change = changelog.parse_commit("3", "docs: expand dataset guide (#3830)", "", "user")
    internal_test = changelog.parse_commit("4", "test(data): add split tests (#3999)", "", "user")

    assert sec is not None
    assert doc_fix is not None
    assert doc_change is not None
    assert internal_test is not None

    assert changelog.classify_commit(sec) == ("Changed", "🔒")
    assert changelog.classify_commit(doc_fix) == ("Fixed", "📚")
    assert changelog.classify_commit(doc_change) == ("Changed", "📚")
    assert changelog.classify_commit(internal_test) is None
    assert changelog.classify_commit(internal_test, include_internal=True) == ("Changed", "🧪")


def test_generate_update_and_extract_identical_roundtrip() -> None:
    """Extracted release notes match the generated CHANGELOG.md section exactly."""
    entries = [
        changelog.parse_commit("1", "feat(model): add VerdictAD model (#3944)", "", "Andersen"),
        changelog.parse_commit("2", "fix(data): restore MVTec autodownloads (#3948)", "", "Ashwin"),
        changelog.parse_commit(
            "3",
            "chore(deps): update dependency pillow to v13 [security] (#3939)",
            "",
            "renovate",
        ),
        changelog.parse_commit("4", "chore(release): restore 2.7.2.dev0 (#3934)", "", "bot"),
    ]
    parsed_entries = [e for e in entries if e is not None]

    section = changelog.build_release_section(
        version="2.7.2",
        release_date="2026-11-26",
        entries=parsed_entries,
        author_lookup={
            3944: "andersendsa",
            3948: "ashwinvaidya17",
            3939: "oep-renovate[bot]",
        },
        use_github_api=False,
    )

    existing = (
        "# Changelog\n\n"
        "## [Unreleased]\n\n"
        "### Fixed\n\n"
        "- 🐞 **data**: Stale unreleased entry\n\n"
        "## [v2.7.1] - 2026-11-16\n\n"
        "### Fixed\n\n"
        "- 🐞 **sampling**: Old fix\n"
    )
    updated = changelog.update_changelog_content(existing, "2.7.2", section)

    assert "Stale unreleased entry" not in updated
    assert "## [v2.7.1] - 2026-11-16" in updated
    assert "- 🐞 **sampling**: Old fix" in updated

    updated_again = changelog.update_changelog_content(updated, "2.7.2", section)
    assert updated_again == updated

    extracted = changelog.extract_release_notes(updated, "lib/v2.7.2")
    expected_body = "\n".join(section.splitlines()[2:]) + "\n"
    assert extracted == expected_body
    assert "## [v2.7.1]" not in extracted


def test_extract_existing_repo_changelog() -> None:
    """Extracts real historical releases from the repo CHANGELOG.md cleanly."""
    repo_changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    notes = changelog.extract_release_notes(repo_changelog, "lib/v2.7.1")
    assert "### Changed" in notes
    assert "### Fixed" in notes
    assert "### Added" not in notes
    assert "### Removed" not in notes
    assert "### Deprecated" not in notes
    assert "## [v2.7.0]" not in notes


def test_extract_missing_or_empty_version_raises() -> None:
    """Fails clearly when a version heading is missing or contains no entries."""
    with pytest.raises(ValueError, match="not found"):
        changelog.extract_release_notes("## [v2.7.1] - 2026-11-16\n\n### Fixed\n\n- Bug\n", "2.9.9")

    with pytest.raises(ValueError, match="has no changelog entries"):
        changelog.extract_release_notes("## [v2.8.0] - 2026-12-01\n\n### Added\n", "2.8.0")


def test_end_to_end_cli_git_repo_replication(tmp_path: Path) -> None:
    """Replicates the full generate -> update -> extract CI workflow in a temp git repo."""
    git_bin = shutil.which("git")
    assert git_bin is not None

    def git(*args: str) -> None:
        subprocess.run(  # noqa: S603
            [git_bin, *args],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )

    git("init")
    git("config", "user.name", "Test Maintainer")
    git("config", "user.email", "maintainer@example.com")

    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\n## [Unreleased]\n\n## [v2.7.1] - 2026-11-16\n\n### Fixed\n\n- Old item\n",
        encoding="utf-8",
    )
    git("add", "CHANGELOG.md")
    git("commit", "-m", "chore(release): prepare 2.7.1 (#3930)")
    git("tag", "lib/v2.7.1")

    for idx, msg in enumerate(
        [
            "chore(release): restore 2.7.2.dev0 (#3934)",
            "fix(model): use math attention during ONNX export (#3936)",
            "🚀 feat(data): add synthetic anomaly preset (#3990)",
        ],
        start=1,
    ):
        (tmp_path / f"file_{idx}.txt").write_text(msg, encoding="utf-8")
        git("add", f"file_{idx}.txt")
        git("commit", "-m", msg)

    gen_res = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(SCRIPT_PATH),
            "generate",
            "--version",
            "2.7.2",
            "--date",
            "2026-11-26",
            "--no-github-api",
            "--update",
            "--changelog",
            "CHANGELOG.md",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Updated CHANGELOG.md with [v2.7.2]" in gen_res.stdout

    release_notes_path = tmp_path / "release-notes.md"
    subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(SCRIPT_PATH),
            "extract",
            "--version",
            "lib/v2.7.2",
            "--changelog",
            "CHANGELOG.md",
            "--output",
            str(release_notes_path),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    changelog_text = (tmp_path / "CHANGELOG.md").read_text(encoding="utf-8")
    notes_text = release_notes_path.read_text(encoding="utf-8")

    assert "## [v2.7.2] - 2026-11-26" in changelog_text
    assert "restore 2.7.2.dev0" not in notes_text
    assert (
        "- 🚀 **data**: Add synthetic anomaly preset in https://github.com/open-edge-platform/anomalib/pull/3990"
        in notes_text
    )
    assert (
        "- 🐞 **model**: Use math attention during ONNX export in https://github.com/open-edge-platform/anomalib/pull/3936"
        in notes_text
    )

    section_start = changelog_text.index("## [v2.7.2] - 2026-11-26\n")
    section_end = changelog_text.index("## [v2.7.1] - 2026-11-16\n")
    section_body = changelog_text[section_start:section_end].split("\n", 1)[1].strip() + "\n"
    assert notes_text == section_body


def test_bump_pyproject_version(tmp_path: Path) -> None:
    """Bumps [project].version in pyproject.toml without altering other sections."""
    pyproject_text = (
        '[build-system]\nrequires = ["hatchling"]\n\n'
        '[project]\nname = "anomalib"\nversion = "2.7.1"\n\n'
        '[tool.ruff]\nversion = "ignore-me"\n'
    )
    updated = changelog.bump_pyproject_version(pyproject_text, "lib/v2.7.2")
    assert 'version = "2.7.2"' in updated
    assert 'version = "ignore-me"' in updated

    with pytest.raises(ValueError, match="Could not find"):
        changelog.bump_pyproject_version('[project]\nname = "anomalib"\n', "2.7.2")

    # Verify CLI --pyproject integration against real git repo in tmp_path
    pyproject_file = tmp_path / "pyproject.toml"
    pyproject_file.write_text(pyproject_text, encoding="utf-8")
    git_bin = shutil.which("git")
    assert git_bin is not None
    subprocess.run([git_bin, "init"], cwd=tmp_path, check=True, capture_output=True)  # noqa: S603
    subprocess.run([git_bin, "config", "user.name", "Test"], cwd=tmp_path, check=True, capture_output=True)  # noqa: S603
    subprocess.run([git_bin, "config", "user.email", "t@example.com"], cwd=tmp_path, check=True, capture_output=True)  # noqa: S603
    subprocess.run([git_bin, "add", "pyproject.toml"], cwd=tmp_path, check=True, capture_output=True)  # noqa: S603
    subprocess.run([git_bin, "commit", "-m", "chore: init"], cwd=tmp_path, check=True, capture_output=True)  # noqa: S603
    subprocess.run([git_bin, "tag", "lib/v2.7.1"], cwd=tmp_path, check=True, capture_output=True)  # noqa: S603
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run([git_bin, "add", "a.py"], cwd=tmp_path, check=True, capture_output=True)  # noqa: S603
    subprocess.run(  # noqa: S603
        [git_bin, "commit", "-m", "fix(core): bump test (#4001)"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(SCRIPT_PATH),
            "generate",
            "--version",
            "2.7.2",
            "--no-github-api",
            "--update",
            "--changelog",
            str(tmp_path / "CHANGELOG.md"),
            "--pyproject",
            str(pyproject_file),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert 'version = "2.7.2"' in pyproject_file.read_text(encoding="utf-8")
    assert "## [v2.7.2]" in (tmp_path / "CHANGELOG.md").read_text(encoding="utf-8")
