# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Test file for the combined path-traversal ruleset
import os
import pathlib
import shutil
from typing import Annotated

from fastapi import Depends, File, HTTPException, Query, UploadFile, status

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


# ---------------------------------------------------------------------------
# Shared helpers — mirror the real service and repository code
# ---------------------------------------------------------------------------


def _assert_within_project(full_path: str, project_folder: str) -> None:
    """Mirrors BinaryRepository._assert_within_project()."""
    project_root = os.path.realpath(project_folder)
    common = os.path.commonpath([project_root, full_path])
    if common != project_root:
        raise ValueError("Invalid filename: path traversal detected")


def _validate_filename(filename: str, expected_folder: str) -> str:
    """Mirrors video_service._validate_filename()."""
    if not filename or os.path.sep in filename or ".." in filename:
        raise ValueError("Invalid filename")
    full_path = os.path.realpath(os.path.join(expected_folder, filename))
    expected_resolved = os.path.realpath(expected_folder)
    if os.path.commonpath([expected_resolved, full_path]) != expected_resolved:
        raise ValueError("Path traversal detected")
    return full_path


# ===========================================================================
# SECTION 1 — path-join-without-realpath-validation
#
# ===========================================================================


def save_file_pre_fix(project_folder: str, filename: str, content: bytes) -> str:
    """Pre-fix: join result used directly — no realpath, no bounds check."""
    full_path = os.path.join(project_folder, filename)  # SOURCE
    folder, _ = full_path.split(filename)
    # ruleid: path-join-without-realpath-validation
    os.makedirs(folder, exist_ok=True)
    # ruleid: path-join-without-realpath-validation
    with open(full_path, "wb") as f:
        f.write(content)
    return full_path


def save_file_post_fix(project_folder: str, filename: str, content: bytes) -> str:
    """Post-fix: realpath resolves traversal before any file op."""
    full_path = os.path.realpath(os.path.join(project_folder, filename))  # SANITIZER
    _assert_within_project(full_path, project_folder)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    # ok: path-join-without-realpath-validation
    with open(full_path, "wb") as f:
        f.write(content)
    return full_path


def write_pathlib_pre_fix(project_folder: str, filename: str, content: bytes) -> None:
    """pathlib.Path.open() is a sink — join result reaches pathlib write."""
    full_path = os.path.join(project_folder, filename)  # SOURCE
    # ruleid: path-join-without-realpath-validation
    with pathlib.Path(full_path).open("wb") as f:
        f.write(content)


def write_pathlib_post_fix(project_folder: str, filename: str, content: bytes) -> None:
    """pathlib.Path.open() safe — realpath applied first."""
    full_path = os.path.realpath(os.path.join(project_folder, filename))  # SANITIZER
    # ok: path-join-without-realpath-validation
    with pathlib.Path(full_path).open("wb") as f:
        f.write(content)


def copy_file_pre_fix(project_folder: str, filename: str, dest: str) -> None:
    """shutil.copy() is a sink — source path from join, no realpath."""
    src = os.path.join(project_folder, filename)  # SOURCE
    # ruleid: path-join-without-realpath-validation
    shutil.copy(src, dest)


def copy_file_post_fix(project_folder: str, filename: str, dest: str) -> None:
    """shutil.copy() safe — realpath applied first."""
    src = os.path.realpath(os.path.join(project_folder, filename))  # SANITIZER
    # ok: path-join-without-realpath-validation
    shutil.copy(src, dest)


def rename_file_pre_fix(project_folder: str, filename: str, new_name: str) -> None:
    """os.rename() is a sink — old_path from join, no realpath."""
    old_path = os.path.join(project_folder, filename)  # SOURCE
    new_path = os.path.join(project_folder, new_name)
    # ruleid: path-join-without-realpath-validation
    os.rename(old_path, new_path)


def rename_file_post_fix(project_folder: str, filename: str, new_name: str) -> None:
    """os.rename() safe — both paths resolved via realpath first."""
    old_path = os.path.realpath(os.path.join(project_folder, filename))  # SANITIZER
    new_path = os.path.realpath(os.path.join(project_folder, new_name))  # SANITIZER
    # ok: path-join-without-realpath-validation
    os.rename(old_path, new_path)

def extract_snapshot_to_path_hardcoded(temp_dir: str) -> None:
    """Hardcoded subdirectory names — second arg is a string literal, not user input."""
    normal_dir = os.path.join(temp_dir, "normal")
    abnormal_dir = os.path.join(temp_dir, "abnormal")
    # ok: path-join-without-realpath-validation
    os.makedirs(normal_dir, exist_ok=True)
    # ok: path-join-without-realpath-validation
    os.makedirs(abnormal_dir, exist_ok=True)


def write_to_hardcoded_subdir(temp_dir: str, content: bytes) -> None:
    """Hardcoded filename — second arg is a string literal."""
    save_path = os.path.join(temp_dir, "output.bin")
    # ok: path-join-without-realpath-validation
    with open(save_path, "wb") as f:
        f.write(content)


# ===========================================================================
# SECTION 2 — binary-repo-get-full-path-without-realpath
#
# ===========================================================================


class BinaryRepositoryPreFix:
    """Pre-fix BinaryRepository: get_full_path() result used without realpath."""

    project_folder_path: str

    def get_full_path(self, filename: str) -> str:
        """Returns os.path.join(project_folder_path, filename) — unchanged."""
        return os.path.join(self.project_folder_path, filename)

    def save_file_pre_fix(self, filename: str, content: bytes) -> str:
        full_path = self.get_full_path(filename)  # SOURCE
        folder, _ = full_path.split(filename)
        # ruleid: binary-repo-get-full-path-without-realpath
        os.makedirs(folder, exist_ok=True)
        # ruleid: binary-repo-get-full-path-without-realpath
        with open(full_path, "wb") as f:
            f.write(content)
        return full_path

    def read_file_pre_fix(self, filename: str) -> bytes:
        full_path = self.get_full_path(filename)  # SOURCE
        if not os.path.isfile(full_path):
            raise FileNotFoundError(f"File not found: {full_path}")
        # ruleid: binary-repo-get-full-path-without-realpath
        with open(full_path, "rb") as fp:
            return fp.read()

    def delete_file_pre_fix(self, filename: str) -> None:
        full_path = self.get_full_path(filename)  # SOURCE
        if os.path.isfile(full_path):
            # ruleid: binary-repo-get-full-path-without-realpath
            os.remove(full_path)

    def write_bytes_pre_fix(self, filename: str, content: bytes) -> None:
        """New sink: pathlib.write_bytes() — no realpath."""
        full_path = self.get_full_path(filename)  # SOURCE
        # ruleid: binary-repo-get-full-path-without-realpath
        pathlib.Path(full_path).write_bytes(content)

    def copy_file_pre_fix(self, filename: str, dest: str) -> None:
        """New sink: shutil.copy() — get_full_path result, no realpath."""
        full_path = self.get_full_path(filename)  # SOURCE
        # ruleid: binary-repo-get-full-path-without-realpath
        shutil.copy(full_path, dest)


class BinaryRepositoryPostFix:
    """Post-fix BinaryRepository: every path goes through realpath first."""

    project_folder_path: str

    def get_full_path(self, filename: str) -> str:
        return os.path.join(self.project_folder_path, filename)

    def _assert_within_project(self, full_path: str) -> None:
        project_root = os.path.realpath(self.project_folder_path)
        common = os.path.commonpath([project_root, full_path])
        if common != project_root:
            raise ValueError("Invalid filename: path traversal detected")

    def save_file_post_fix(self, filename: str, content: bytes) -> str:
        full_path = os.path.realpath(self.get_full_path(filename))  # SANITIZER
        self._assert_within_project(full_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        # ok: binary-repo-get-full-path-without-realpath
        with open(full_path, "wb") as f:
            f.write(content)
        return full_path

    def read_file_post_fix(self, filename: str) -> bytes:
        full_path = os.path.realpath(self.get_full_path(filename))  # SANITIZER
        self._assert_within_project(full_path)
        if not os.path.isfile(full_path):
            raise FileNotFoundError(f"File not found: {full_path}")
        # ok: binary-repo-get-full-path-without-realpath
        with open(full_path, "rb") as fp:
            return fp.read()

    def delete_file_post_fix(self, filename: str) -> None:
        full_path = os.path.realpath(self.get_full_path(filename))  # SANITIZER
        self._assert_within_project(full_path)
        if os.path.isfile(full_path):
            # ok: binary-repo-get-full-path-without-realpath
            os.remove(full_path)

    def write_bytes_post_fix(self, filename: str, content: bytes) -> None:
        """New sink: pathlib.write_bytes() safe after realpath."""
        full_path = os.path.realpath(self.get_full_path(filename))  # SANITIZER
        self._assert_within_project(full_path)
        # ok: binary-repo-get-full-path-without-realpath
        pathlib.Path(full_path).write_bytes(content)


# ===========================================================================
# SECTION 3 — fastapi-upload-filename-path-traversal
#
# ===========================================================================


# Direct UploadFile — join → open.
def upload_direct_open_pre_fix(file: UploadFile = File(...)) -> None:
    if file.filename is None or "." not in file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No extension")
    path = os.path.join("/data/uploads", file.filename)
    # ruleid: fastapi-upload-filename-path-traversal, path-join-without-realpath-validation
    with open(path, "wb") as f:
        f.write(b"")


# Direct UploadFile — makedirs sink.
def upload_direct_makedirs_pre_fix(file: UploadFile = File(...), base_dir: str = "/data") -> None:
    path = os.path.join(base_dir, file.filename)
    # ruleid: fastapi-upload-filename-path-traversal, path-join-without-realpath-validation
    os.makedirs(path, exist_ok=True)


# Direct UploadFile — os.remove sink.
def upload_direct_remove_pre_fix(file: UploadFile = File(...), base_dir: str = "/data") -> None:
    path = os.path.join(base_dir, file.filename)
    # ruleid: fastapi-upload-filename-path-traversal, path-join-without-realpath-validation
    os.remove(path)


# Direct UploadFile — shutil.rmtree sink.
def upload_direct_rmtree_pre_fix(file: UploadFile = File(...), base_dir: str = "/data") -> None:
    path = os.path.join(base_dir, file.filename)
    # ruleid: fastapi-upload-filename-path-traversal, path-join-without-realpath-validation
    shutil.rmtree(path)


# Annotated UploadFile — intermediate variable → makedirs + open.
async def upload_annotated_pre_fix(
    file: Annotated[UploadFile, Depends(upload_direct_open_pre_fix)],
    base_dir: str = "/data",
) -> None:
    filename = file.filename
    full_path = os.path.join(base_dir, filename)
    folder = os.path.dirname(full_path)
    # ruleid: fastapi-upload-filename-path-traversal, path-join-without-realpath-validation
    os.makedirs(folder, exist_ok=True)
    # ruleid: fastapi-upload-filename-path-traversal, path-join-without-realpath-validation
    open(full_path, "wb")


# pathlib.Path.open().
def upload_pathlib_pre_fix(file: UploadFile = File(...), base_dir: str = "/data") -> None:
    """pathlib.Path.open() is now a sink — UploadFile.filename not validated."""
    full_path = os.path.join(base_dir, file.filename)
    # ruleid: fastapi-upload-filename-path-traversal, path-join-without-realpath-validation
    with pathlib.Path(full_path).open("wb") as f:
        f.write(b"")


# shutil.copy().
def upload_shutil_copy_pre_fix(file: UploadFile = File(...), tmp: str = "/tmp/uploads") -> None:
    """shutil.copy() is now a sink — tainted filename in source argument."""
    src = os.path.join("/data/uploads", file.filename)
    # ruleid: fastapi-upload-filename-path-traversal, path-join-without-realpath-validation
    shutil.copy(src, tmp)


# Post-fix — _validate_filename returns the safe path.
def upload_post_fix_validate(file: UploadFile = File(...), base_dir: str = "/data") -> None:
    safe_path = _validate_filename(file.filename, base_dir)  # SANITIZER
    # ok: fastapi-upload-filename-path-traversal
    with open(safe_path, "wb") as f:
        f.write(b"")


# Hardcoded filename — no user input, no taint source.
def write_hardcoded() -> None:
    # ok: fastapi-upload-filename-path-traversal
    open("/data/output.mp4", "wb")


# ===========================================================================
# SECTION 4 — fastapi-query-param-filename-path-traversal
#
# ===========================================================================


# Annotated[str, Query] — join → os.remove.
async def delete_file_query_pre_fix(
    filename: Annotated[str, Query(description="Filename to delete")],
    base_dir: str = "/data/videos",
) -> None:
    """No _validate_filename call — Query param drives os.remove directly."""
    path = os.path.join(base_dir, filename)
    # ruleid: fastapi-query-param-filename-path-traversal, path-join-without-realpath-validation
    os.remove(path)


# str = Query(...) form — join → open.
async def read_file_query_pre_fix(
    filename: str = Query(description="Filename to read"),
    base_dir: str = "/data",
) -> bytes:
    """No validation — Query param drives open() directly."""
    path = os.path.join(base_dir, filename)
    # ruleid: fastapi-query-param-filename-path-traversal, path-join-without-realpath-validation
    with open(path, "rb") as f:
        return f.read()


# Annotated[str, Query] — pathlib.write_bytes sink.
async def write_file_query_pathlib_pre_fix(
    filename: Annotated[str, Query(description="Output filename")],
    content: bytes = b"",
    base_dir: str = "/data",
) -> None:
    """pathlib.write_bytes() is a sink — Query param not validated."""
    full_path = os.path.join(base_dir, filename)
    # ruleid: fastapi-query-param-filename-path-traversal, path-join-without-realpath-validation
    pathlib.Path(full_path).write_bytes(content)


# Annotated[str, Query] — _validate_filename sanitizes before delete.
async def delete_file_query_post_fix(
    filename: Annotated[str, Query(description="Filename to delete")],
    base_dir: str = "/data/videos",
) -> None:
    """_validate_filename() returns the safe, resolved path."""
    safe_path = _validate_filename(filename, base_dir)  # SANITIZER
    # ok: fastapi-query-param-filename-path-traversal
    os.remove(safe_path)
