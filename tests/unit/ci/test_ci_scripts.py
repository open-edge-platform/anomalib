# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Compact unit and round-trip tests for `.github/scripts/{changelog,check_pr_title}.py`."""

from __future__ import annotations

import http.client
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType

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


def test_parse_classify_and_normalize() -> None:
    """Covers version normalization, commit parsing, housekeeping filtering, and section mapping."""
    assert changelog.normalize_version("lib/v2.7.1") == "2.7.1"
    with pytest.raises(ValueError, match="Invalid empty version"):
        changelog.normalize_version("lib/v")

    breaking = changelog.parse_commit(
        "1",
        "🐞 chore(model)!: remove legacy exporter (#3826)",
        "BREAKING CHANGE: dynamo=False removed.",
        "Maintainer",
    )
    assert breaking is not None
    assert breaking.pr_number == 3826
    assert changelog.classify_commit(breaking) == ("Removed", "⚠️")

    sec = changelog.parse_commit("2", "chore(deps): update pillow [security] (#3939)", "", "bot")
    doc_fix = changelog.parse_commit("3", "docs(cli): clarify usage (#3931)", "", "bot")
    internal = changelog.parse_commit("4", "test(data): add split test (#3999)", "", "user")
    housekeeping = changelog.parse_commit("5", "chore(release): restore 2.7.2.dev0 (#3934)", "", "bot")
    docs_housekeeping = changelog.parse_commit("6", "docs(changelog): prepare CHANGELOG.md for v2.7.2 (#3950)", "", "b")
    reverted = changelog.parse_commit("7", "⏪ revert(cli): restore legacy flag (#3955)", "", "user")
    assert sec is not None
    assert doc_fix is not None
    assert internal is not None
    assert housekeeping is not None
    assert docs_housekeeping is not None
    assert reverted is not None

    assert changelog.classify_commit(sec) == ("Changed", "🔒")
    assert changelog.classify_commit(doc_fix) == ("Fixed", "📚")
    assert changelog.classify_commit(internal) is None
    assert changelog.classify_commit(internal, include_internal=True) == ("Changed", "🧪")
    assert changelog.classify_commit(housekeeping) is None
    assert changelog.classify_commit(docs_housekeeping) is None
    assert changelog.classify_commit(reverted) == ("Changed", "⏪")


def test_fetch_pr_author(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validates repo slug checks, bot login normalization, and HTTP error fallback."""
    with pytest.raises(ValueError, match="Invalid GitHub repository slug"):
        changelog.fetch_pr_author("../evil/repo", 1)

    class _FakeResponse:
        def __init__(self, status: int, body: bytes) -> None:
            self.status = status
            self._body = body

        def read(self) -> bytes:
            return self._body

    responses = [
        _FakeResponse(200, json.dumps({"user": {"login": "app/oep-renovate"}}).encode()),
        _FakeResponse(404, b'{"message": "Not Found"}'),
    ]

    class _FakeHTTPSConnection:
        def __init__(self, host: str, timeout: int = 15) -> None:
            assert host == "api.github.com"
            self.timeout = timeout
            self.closed = False
            self.auth_header: str | None = None

        def request(self, method: str, path: str, headers: dict[str, str]) -> None:
            assert method == "GET"
            assert path.startswith("/repos/open-edge-platform/anomalib/pulls/")
            assert "User-Agent" in headers
            self.auth_header = headers.get("Authorization")

        def getresponse(self) -> _FakeResponse:
            assert not self.closed
            return responses.pop(0)

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(http.client, "HTTPSConnection", _FakeHTTPSConnection)
    assert (
        changelog.fetch_pr_author("open-edge-platform/anomalib", 3954, token="dummy")  # noqa: S106
        == "oep-renovate[bot]"
    )
    assert changelog.fetch_pr_author("open-edge-platform/anomalib", 9999) is None


def test_changelog_generate_bump_and_extract_roundtrip(tmp_path: Path) -> None:
    """Replicates generate -> pyproject bump -> idempotent update -> identical extract in git."""
    git_bin = shutil.which("git")
    assert git_bin is not None

    def git(*args: str) -> None:
        subprocess.run(  # noqa: S603
            [git_bin, "-c", "commit.gpgsign=false", *args],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )

    git("init")
    git("config", "user.name", "Maintainer")
    git("config", "user.email", "maintainer@example.com")

    changelog_file = tmp_path / "CHANGELOG.md"
    pyproject_file = tmp_path / "pyproject.toml"
    changelog_file.write_text(
        "# Changelog\n\n## [Unreleased]\n\n- Old unreleased\n\n## [v2.7.1] - 2026-11-16\n\n### Fixed\n\n- Old\n",
        encoding="utf-8",
    )
    pyproject_file.write_text(
        '[project]\nname = "anomalib"\nversion = "2.7.1"\n\n[tool.ruff]\nversion = "keep"\n',
        encoding="utf-8",
    )
    git("add", "CHANGELOG.md", "pyproject.toml")
    git("commit", "-m", "chore(release): prepare 2.7.1 (#3930)")
    git("tag", "lib/v2.7.1")

    for i, msg in enumerate(
        [
            "chore(release): restore 2.7.2.dev0 (#3934)",
            "🚀 feat(data): add dataset preset (#3990)",
            "fix(model): use math attention (#3936)",
        ],
        start=1,
    ):
        (tmp_path / f"f{i}.txt").write_text(msg, encoding="utf-8")
        git("add", f"f{i}.txt")
        git("commit", "-m", msg)

    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "changelog.py"),
        "generate",
        "--version",
        "2.7.2",
        "--date",
        "2026-11-26",
        "--no-github-api",
        "--update",
        "--changelog",
        str(changelog_file),
        "--pyproject",
        str(pyproject_file),
    ]
    subprocess.run(cmd, cwd=tmp_path, check=True, capture_output=True, text=True)  # noqa: S603
    first = changelog_file.read_text(encoding="utf-8")
    subprocess.run(cmd, cwd=tmp_path, check=True, capture_output=True, text=True)  # noqa: S603
    assert changelog_file.read_text(encoding="utf-8") == first  # idempotent

    pyproject_text = pyproject_file.read_text(encoding="utf-8")
    assert 'version = "2.7.2"' in pyproject_text
    assert 'version = "keep"' in pyproject_text
    assert "Old unreleased" not in first
    assert "restore 2.7.2.dev0" not in first

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
    assert "- 🚀 **data**: Add dataset preset in https://github.com/open-edge-platform/anomalib/pull/3990" in notes
    assert "- 🐞 **model**: Use math attention in https://github.com/open-edge-platform/anomalib/pull/3936" in notes


def test_extract_repo_changelog_and_errors() -> None:
    """Extracts a real repo version cleanly and rejects missing/empty sections."""
    repo_changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    notes = changelog.extract_release_notes(repo_changelog, "lib/v2.7.1")
    assert "### Changed" in notes
    assert "### Fixed" in notes
    assert "### Added" not in notes

    with pytest.raises(ValueError, match="not found"):
        changelog.extract_release_notes(repo_changelog, "9.9.9")
    with pytest.raises(ValueError, match="has no changelog entries"):
        changelog.extract_release_notes("## [v2.8.0] - 2026-12-01\n\n### Added\n", "2.8.0")
    with pytest.raises(ValueError, match="Could not find"):
        changelog.bump_pyproject_version('[project]\nname = "anomalib"\n', "2.7.2")

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
