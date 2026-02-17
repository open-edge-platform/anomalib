# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for anomalib.utils.path (create_versioned_dir, resolve_versioned_path)."""

import sys
from pathlib import Path

from anomalib.utils.path import create_versioned_dir, resolve_versioned_path


class TestResolveVersionedPath:
    """Tests for resolve_versioned_path."""

    @staticmethod
    def test_path_without_latest_unchanged(tmp_path: Path) -> None:
        """Path with no 'latest' component is returned unchanged."""
        p = tmp_path / "some" / "path" / "file.ckpt"
        p.parent.mkdir(parents=True, exist_ok=True)
        assert resolve_versioned_path(p) == p
        assert resolve_versioned_path(str(p)) == p

    @staticmethod
    def test_path_with_latest_replaced_when_version_dirs_exist(tmp_path: Path) -> None:
        """When parent of 'latest' contains v0, v1, ..., 'latest' is replaced with highest vN."""
        root = tmp_path / "exp"
        (root / "v0").mkdir(parents=True)
        (root / "v1").mkdir()
        (root / "v2").mkdir()
        # On POSIX, resolve_versioned_path follows the symlink; create it so resolve() works
        if sys.platform != "win32":
            (root / "latest").symlink_to(root / "v2", target_is_directory=True)
        path_with_latest = root / "latest" / "weights" / "model.ckpt"
        (root / "v2" / "weights").mkdir(parents=True)
        resolved = resolve_versioned_path(path_with_latest)
        assert resolved == root / "v2" / "weights" / "model.ckpt"

    @staticmethod
    def test_path_with_latest_no_version_dirs_unchanged(tmp_path: Path) -> None:
        """When parent of 'latest' has no vN dirs, path is returned unchanged."""
        root = tmp_path / "exp"
        root.mkdir(parents=True)
        path_with_latest = root / "latest" / "weights" / "model.ckpt"
        resolved = resolve_versioned_path(path_with_latest)
        assert resolved == path_with_latest

    @staticmethod
    def test_path_with_latest_parent_missing_unchanged(tmp_path: Path) -> None:
        """When parent of 'latest' does not exist, path is returned unchanged."""
        path_with_latest = tmp_path / "nonexistent" / "latest" / "model.ckpt"
        resolved = resolve_versioned_path(path_with_latest)
        assert resolved == path_with_latest

    @staticmethod
    def test_path_with_latest_only_v0(tmp_path: Path) -> None:
        """When only v0 exists, 'latest' is replaced with v0."""
        root = tmp_path / "exp"
        (root / "v0").mkdir(parents=True)
        if sys.platform != "win32":
            (root / "latest").symlink_to(root / "v0", target_is_directory=True)
        path_with_latest = root / "latest" / "model.ckpt"
        resolved = resolve_versioned_path(path_with_latest)
        assert resolved == root / "v0" / "model.ckpt"

    @staticmethod
    def test_first_occurrence_of_latest_replaced(tmp_path: Path) -> None:
        """Only the first occurrence of 'latest' in path parts is replaced."""
        root = tmp_path / "exp"
        (root / "v1").mkdir(parents=True)
        if sys.platform != "win32":
            (root / "latest").symlink_to(root / "v1", target_is_directory=True)
        # path like .../latest/latest/file - first 'latest' resolved, second stays
        path_with_two = root / "latest" / "latest" / "file.ckpt"
        resolved = resolve_versioned_path(path_with_two)
        assert resolved == root / "v1" / "latest" / "file.ckpt"


class TestCreateVersionedDir:
    """Tests for create_versioned_dir (return value is real path)."""

    @staticmethod
    def test_returns_real_version_path_not_latest(tmp_path: Path) -> None:
        """create_versioned_dir returns the real version directory path, not the 'latest' link."""
        root = tmp_path / "results"
        result = create_versioned_dir(root)
        assert result.name == "v0"
        assert result == root / "v0"
        assert (root / "latest").exists()
        # Returned path must not be the link path
        assert result != root / "latest"

    @staticmethod
    def test_increments_version(tmp_path: Path) -> None:
        """Second call returns v1, etc."""
        root = tmp_path / "results"
        r1 = create_versioned_dir(root)
        r2 = create_versioned_dir(root)
        assert r1.name == "v0"
        assert r2.name == "v1"
        assert r1 != r2
