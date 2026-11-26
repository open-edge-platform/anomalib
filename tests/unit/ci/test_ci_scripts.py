# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Compact unit tests for `.github/scripts/{changelog,check_pr_title}.py` and `[tool.git-cliff]`."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib  # type: ignore[no-redef]

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / ".github" / "scripts"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"anomalib_ci_{name}", SCRIPTS_DIR / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


changelog = _load("changelog")
check_pr_title = _load("check_pr_title")


def test_cliff_config_is_valid_toml() -> None:
    """Verifies `[tool.git-cliff]` in `pyproject.toml` parses cleanly and targets `lib/v*` tags."""
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["git-cliff"]
    assert config["remote"]["github"] == {"owner": "open-edge-platform", "repo": "anomalib"}
    assert "lib/v" in config["git"]["tag_pattern"]
    assert config["git"]["conventional_commits"] is True
    assert config["changelog"]["body"]


def test_changelog_update_and_extract_roundtrip(tmp_path: Path) -> None:
    """Verifies idempotent `update` and verbatim `extract` via the CLI."""
    assert changelog.normalize_version("lib/v2.7.1") == "2.7.1"
    with pytest.raises(ValueError, match="Invalid empty version"):
        changelog.normalize_version("lib/v")

    changelog_file = tmp_path / "CHANGELOG.md"
    changelog_file.write_text(
        "# Changelog\n\n## [Unreleased]\n\n- Old unreleased\n\n## [v2.7.1] - 2026-11-16\n\n### Fixed\n\n- Old\n",
        encoding="utf-8",
    )
    section_file = tmp_path / "section.md"
    section_file.write_text(
        "## [v2.7.2] - 2026-11-26\n\n"
        "### Added\n\n"
        "- 🚀 **data**: Add dataset preset in https://github.com/open-edge-platform/anomalib/pull/3990\n\n"
        "### Fixed\n\n"
        "- 🐞 **model**: Use math attention in https://github.com/open-edge-platform/anomalib/pull/3936\n",
        encoding="utf-8",
    )

    update_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "changelog.py"),
        "update",
        "--version",
        "2.7.2",
        "--input",
        str(section_file),
        "--changelog",
        str(changelog_file),
    ]
    subprocess.run(update_cmd, cwd=tmp_path, check=True, capture_output=True, text=True)  # noqa: S603
    first = changelog_file.read_text(encoding="utf-8")
    subprocess.run(update_cmd, cwd=tmp_path, check=True, capture_output=True, text=True)  # noqa: S603
    assert changelog_file.read_text(encoding="utf-8") == first  # idempotent
    assert "Old unreleased" not in first

    notes_file = tmp_path / "notes.md"
    subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(SCRIPTS_DIR / "changelog.py"),
            "extract",
            "--version",
            "lib/v2.7.2",
            "--changelog",
            str(changelog_file),
            "--output",
            str(notes_file),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    notes = notes_file.read_text(encoding="utf-8")
    body = first[first.index("## [v2.7.2]") : first.index("## [v2.7.1]")].split("\n", 1)[1].strip() + "\n"
    assert notes == body


def test_extract_repo_changelog_and_rich_markdown() -> None:
    """Extracts a real repo version cleanly, preserves migration markdown, and rejects empty sections."""
    repo_changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    notes = changelog.extract_release_notes(repo_changelog, "lib/v2.7.1")
    assert "### Changed" in notes
    assert "### Fixed" in notes
    assert "### Added" not in notes

    rich_changelog = (
        "## [v2.8.0] - 2026-12-01\n\n"
        "### Added\n\n"
        "### Changed\n\n"
        "- ⚠️ **breaking**: Drop legacy exporter\n\n"
        "#### Migration\n\n"
        "First paragraph.\n\n"
        "```python\n"
        "a = 1\n\n"
        "b = 2\n"
        "```\n"
    )
    extracted_rich = changelog.extract_release_notes(rich_changelog, "2.8.0")
    assert "### Added" not in extracted_rich
    assert "First paragraph.\n\n```python\na = 1\n\nb = 2\n```" in extracted_rich

    with pytest.raises(ValueError, match="not found"):
        changelog.extract_release_notes(repo_changelog, "9.9.9")
    with pytest.raises(ValueError, match="has no changelog entries"):
        changelog.extract_release_notes("## [v2.8.0] - 2026-12-01\n\n### Added\n", "2.8.0")


def test_check_pr_title(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validates Conventional Commit PR titles with optional leading emojis and env fallback."""
    for valid in (
        "feat(model): add VerdictAD",
        "🐞 fix(post_processing): explicit None check",
        "📦 build!: require Python 3.11",
        "⏪ revert(cli): restore parser",
    ):
        assert check_pr_title.main([valid]) == 0

    for invalid in ("", "Update readme", "unknown(data): x", "feat(data): ", "Feat(data): caps"):
        assert check_pr_title.main([invalid]) == 1

    monkeypatch.setenv("PR_TITLE", "🧪 test(ci): verify env var")
    assert check_pr_title.main([]) == 0
