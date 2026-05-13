# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the metadata module."""

import json
from pathlib import Path

import pytest

from anomalib.deploy.metadata import SchemaValidationError, dump_metadata, load_metadata
from anomalib.deploy.metadata.migration import CURRENT_SCHEMA_VERSION, upgrade_to_latest


def _make_valid_metadata() -> dict:
    return {
        "schema_version": "1.0",
        "anomalib_version": "2.1.0",
        "model": "Patchcore",
        "preprocess": [],
        "postprocess": {
            "image_sensitivity": 0.5,
            "pixel_sensitivity": 0.5,
        },
    }


class TestMetadataLoadDump:
    """Tests for metadata load and dump operations."""

    @staticmethod
    def test_load_metadata(tmp_path: Path) -> None:
        """Verify load_metadata reads, upgrades, and validates correctly."""
        metadata = _make_valid_metadata()
        path = tmp_path / "metadata.json"
        path.write_text(json.dumps(metadata))

        loaded = load_metadata(path)
        assert loaded["schema_version"] == "1.0"
        assert loaded["model"] == "Patchcore"
        assert loaded["postprocess"]["image_sensitivity"] == pytest.approx(0.5)

    @staticmethod
    def test_dump_metadata(tmp_path: Path) -> None:
        """Verify dump_metadata writes valid JSON."""
        metadata = _make_valid_metadata()
        path = tmp_path / "metadata.json"
        dump_metadata(metadata, path)

        raw = json.loads(path.read_text())
        assert raw["schema_version"] == "1.0"
        assert raw["model"] == "Patchcore"

    @staticmethod
    def test_validate_missing_fields(tmp_path: Path) -> None:
        """Verify SchemaValidationError raised on incomplete metadata."""
        path = tmp_path / "metadata.json"
        path.write_text(json.dumps({"schema_version": "1.0"}))

        with pytest.raises(SchemaValidationError, match="missing required fields"):
            load_metadata(path)


class TestMigration:
    """Tests for metadata schema migration."""

    @staticmethod
    def test_upgrade_current_version() -> None:
        """Verify current-version metadata passes through unchanged."""
        metadata = _make_valid_metadata()
        result = upgrade_to_latest(metadata)
        assert result == metadata

    @staticmethod
    def test_legacy_to_v1() -> None:
        """Verify legacy metadata (no schema_version) upgrades to v1."""
        legacy = {"model": "Padim"}
        result = upgrade_to_latest(legacy)
        assert result["schema_version"] == "1.0"
        assert result["model"] == "Padim"
        assert result["postprocess"]["image_sensitivity"] == pytest.approx(0.5)

    @staticmethod
    def test_future_version_no_crash() -> None:
        """Verify future schema versions pass through without error."""
        metadata = _make_valid_metadata()
        metadata["schema_version"] = "99.0"
        result = upgrade_to_latest(metadata)
        assert result["schema_version"] == "99.0"

    @staticmethod
    def test_current_schema_version() -> None:
        """Verify CURRENT_SCHEMA_VERSION constant."""
        assert CURRENT_SCHEMA_VERSION == "1.0"
