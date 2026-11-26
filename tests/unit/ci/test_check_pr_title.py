# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for `.github/scripts/check_pr_title.py`."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / ".github" / "scripts" / "check_pr_title.py"


def _load_check_pr_title_module() -> ModuleType:
    """Load `.github/scripts/check_pr_title.py` as a module."""
    spec = importlib.util.spec_from_file_location("anomalib_ci_check_pr_title", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


check_pr_title = _load_check_pr_title_module()


@pytest.mark.parametrize(
    "title",
    [
        "feat(model): add VerdictAD anomaly detection",
        "fix(data): accept valid split ratios",
        "docs: correct OpenSSF badge attribution",
        "🐞 fix(post_processing): use explicit None check",
        "🚀 feat(data): add MAD-Sim dataset",
        "📦 build!: require Python 3.11",
        "chore(deps): update dependency pillow to v13 [security]",
        "revert(cli): restore previous parser behavior",
    ],
)
def test_valid_pr_titles(title: str) -> None:
    """Accepts valid Conventional Commit PR titles with or without leading emojis."""
    assert check_pr_title.main([title]) == 0


@pytest.mark.parametrize(
    "title",
    [
        "",
        "   ",
        "Update readme",
        "unknown(data): something",
        "feat(data): ",
        "Feat(data): capitalized type",
    ],
)
def test_invalid_pr_titles(title: str) -> None:
    """Rejects empty, non-conventional, or unknown-type PR titles."""
    assert check_pr_title.main([title]) == 1


def test_reads_pr_title_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reads PR_TITLE from environment when no positional argument is passed."""
    monkeypatch.setenv("PR_TITLE", "🧪 test(ci): verify PR title check")
    assert check_pr_title.main([]) == 0
