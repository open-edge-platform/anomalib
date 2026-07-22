# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for path traversal in BinaryRepository.

Verifies that save_file, read_file, and delete_file refuse to operate on
filenames that resolve outside the project directory boundary.
"""

import asyncio
import os
import tempfile
from unittest.mock import patch

import pytest

from repositories.binary_repo import VideoBinaryRepository


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TRAVERSAL_FILENAMES = [
    "../../../../tmp/pwned.mp4",
    "../sibling_project/video.mp4",
    "subdir/../../outside.mp4",
    # URL-encoded form should not bypass os.path.realpath, but test explicitly
    "%2e%2e%2foutside.mp4",
    # Absolute paths
    "/tmp/absolute.mp4",
    # Windows-style (relevant if tested on Windows CI)
    "..\\..\\outside.mp4",
]

SAFE_FILENAME = "legitimate_video.mp4"


@pytest.fixture
def fxt_project_dir(tmp_path):
    """A real temporary project directory with the expected layout."""
    project_dir = tmp_path / "projects" / "test_project_id"
    project_dir.mkdir(parents=True)
    return project_dir


@pytest.fixture
def fxt_video_repo(fxt_project_dir):
    """A VideoBinaryRepository whose project_folder_path points at the temp dir."""
    repo = VideoBinaryRepository(project_id="test_project_id")
    # Override the cached property so it uses our isolated temp directory.
    with patch.object(
        type(repo),
        "project_folder_path",
        new_callable=lambda: property(lambda self: str(fxt_project_dir)),
    ):
        yield repo


def run(coro):
    """Run a coroutine synchronously (avoids asyncio.run nesting issues)."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# BinaryRepository.save_file — path traversal must be rejected
# ---------------------------------------------------------------------------


class TestSaveFilePathTraversal:
    """save_file must raise ValueError for any filename that escapes the project dir."""

    @pytest.mark.parametrize("filename", TRAVERSAL_FILENAMES)
    def test_save_file_rejects_traversal(self, tmp_path, filename):
        """Traversal filename must raise ValueError, never write to disk."""
        project_dir = tmp_path / "projects" / "proj"
        project_dir.mkdir(parents=True)
        repo = VideoBinaryRepository(project_id="proj")

        with patch.object(type(repo), "project_folder_path", new_callable=lambda: property(lambda self: str(project_dir))):
            with pytest.raises((ValueError, OSError)):
                run(repo.save_file(filename=filename, content=b"PWNED"))

        # Confirm nothing was written outside the project dir
        assert not (tmp_path / "pwned.mp4").exists()
        assert not (tmp_path / "outside.mp4").exists()

    def test_save_file_accepts_safe_filename(self, tmp_path):
        """A clean filename must be written inside the project directory."""
        project_dir = tmp_path / "projects" / "proj"
        project_dir.mkdir(parents=True)
        repo = VideoBinaryRepository(project_id="proj")

        with patch.object(type(repo), "project_folder_path", new_callable=lambda: property(lambda self: str(project_dir))):
            saved = run(repo.save_file(filename=SAFE_FILENAME, content=b"OK"))

        assert os.path.isfile(saved)
        assert saved.startswith(str(project_dir))

    def test_save_file_bad_payload(self, tmp_path):
        """Malformed payload must be blocked."""
        project_dir = tmp_path / "data" / "videos" / "projects" / "VfQiLbCy9ERHSyRuNpH2mB"
        project_dir.mkdir(parents=True)
        repo = VideoBinaryRepository(project_id="VfQiLbCy9ERHSyRuNpH2mB")

        traversal = "../../../../../../../../tmp/anomalib_pwned.mp4"
        with patch.object(type(repo), "project_folder_path", new_callable=lambda: property(lambda self: str(project_dir))):
            with pytest.raises((ValueError, OSError)):
                run(repo.save_file(filename=traversal, content=b"PWNED"))

        assert not (tmp_path / "tmp" / "anomalib_pwned.mp4").exists()


# ---------------------------------------------------------------------------
# BinaryRepository.read_file — path traversal must be rejected
# ---------------------------------------------------------------------------


class TestReadFilePathTraversal:
    """read_file must not serve files outside the project dir."""

    @pytest.mark.parametrize("filename", TRAVERSAL_FILENAMES)
    def test_read_file_rejects_traversal(self, tmp_path, filename):
        """Traversal filename must raise ValueError (or FileNotFoundError after rejection)."""
        project_dir = tmp_path / "projects" / "proj"
        project_dir.mkdir(parents=True)
        # Plant a target file outside the project dir to make the traversal realistic
        (tmp_path / "secret.mp4").write_bytes(b"SECRET")

        repo = VideoBinaryRepository(project_id="proj")
        with patch.object(type(repo), "project_folder_path", new_callable=lambda: property(lambda self: str(project_dir))):
            with pytest.raises((ValueError, OSError, FileNotFoundError)):
                run(repo.read_file(filename=filename))


# ---------------------------------------------------------------------------
# BinaryRepository.delete_file — path traversal must be rejected
# ---------------------------------------------------------------------------


class TestDeleteFilePathTraversal:
    """delete_file must not remove files outside the project dir."""

    @pytest.mark.parametrize("filename", TRAVERSAL_FILENAMES)
    def test_delete_file_rejects_traversal(self, tmp_path, filename):
        """Traversal filename must raise ValueError; target file must survive."""
        project_dir = tmp_path / "projects" / "proj"
        project_dir.mkdir(parents=True)
        target = tmp_path / "important.mp4"
        target.write_bytes(b"KEEP ME")

        repo = VideoBinaryRepository(project_id="proj")
        with patch.object(type(repo), "project_folder_path", new_callable=lambda: property(lambda self: str(project_dir))):
            with pytest.raises((ValueError, OSError, FileNotFoundError)):
                run(repo.delete_file(filename=filename))

        # The file outside the project must remain untouched
        assert target.exists()
