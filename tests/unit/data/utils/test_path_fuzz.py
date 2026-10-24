# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Check that the fuzzing oracle accepts valid paths and detects security violations."""

import os
from pathlib import Path

import pytest

from tests.fuzz.fuzz_path import fuzz_path, path_utils


@pytest.fixture
def dataset_root(tmp_path: Path) -> Path:
    """Create a resolved dataset directory for oracle checks."""
    root = tmp_path.resolve() / "dataset"
    root.mkdir()
    return root


@pytest.mark.parametrize("data", [b"", b"images/image.png", b"../outside.png", b"\x00", b"\xff", b"a" * 1024])
def test_fuzzer_accepts_expected_outcomes(dataset_root: Path, data: bytes) -> None:
    """Accept valid paths and expected input rejections without requiring Atheris."""
    fuzz_path(data, dataset_root)


def test_fuzzer_rejects_symlink_escape(dataset_root: Path) -> None:
    """Require symlink escapes to fail on platforms with symlink privileges."""
    outside = dataset_root.parent / "outside"
    outside.mkdir()
    try:
        (dataset_root / "escape").symlink_to(outside, target_is_directory=True)
    except OSError as error:
        if os.name == "nt" and error.winerror == 1314:
            pytest.skip("Windows requires Developer Mode or symlink privileges")
        raise
    fuzz_path(b"escape/image.png", dataset_root)


def test_fuzzer_detects_escape(dataset_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Prove the oracle detects a containment check that accepts escapes."""
    monkeypatch.setattr(path_utils, "is_within_directory", lambda *_: True)
    with pytest.raises(AssertionError):
        fuzz_path(b"../outside.png", dataset_root)


def test_fuzzer_detects_unresolved_escape(dataset_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Detect an escape even if the resolver returns an unnormalized path."""
    resolve = path_utils.resolve_path_under_root
    traversal = "images/../../outside.png"

    def unresolved(root: Path, path: str | Path, **kwargs: object) -> Path:
        if path == traversal:
            return root / path
        return resolve(root, path, **kwargs)

    monkeypatch.setattr(path_utils, "resolve_path_under_root", unresolved)
    with pytest.raises(AssertionError):
        fuzz_path(traversal.encode(), dataset_root)


def test_fuzzer_requires_valid_paths(dataset_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Prove the oracle rejects an implementation that denies every path."""

    def reject(*_args: object, **_kwargs: object) -> Path:
        msg = "Rejected every path"
        raise ValueError(msg)

    monkeypatch.setattr(path_utils, "resolve_path_under_root", reject)
    with pytest.raises(ValueError, match="Rejected every path"):
        fuzz_path(b"image.png", dataset_root)
