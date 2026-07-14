# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Metadata loading, dumping, and validation for exported anomalib models."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from anomalib.deploy.metadata.migration import upgrade_to_latest

if TYPE_CHECKING:
    from pathlib import Path


class SchemaValidationError(ValueError):
    """Raised when metadata fails schema validation."""


def load_metadata(path: Path) -> dict:
    """Read metadata.json, auto-upgrade to current schema, and validate.

    Args:
        path (Path): Path to the metadata JSON file.

    Returns:
        dict: Validated and upgraded metadata dictionary.

    Raises:
        SchemaValidationError: If required fields are missing.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    metadata = upgrade_to_latest(raw)
    _validate(metadata)
    return metadata


def dump_metadata(metadata: dict, path: Path) -> None:
    """Validate and write metadata.json.

    Args:
        metadata (dict): Metadata dictionary to write.
        path (Path): Destination file path.

    Raises:
        SchemaValidationError: If required fields are missing.
    """
    _validate(metadata)
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _validate(metadata: dict) -> None:
    """Validate metadata against expected schema fields.

    Args:
        metadata (dict): Metadata dictionary to validate.

    Raises:
        SchemaValidationError: If required fields are missing.
    """
    required = {"schema_version", "anomalib_version", "model", "postprocess"}
    missing = required - metadata.keys()
    if missing:
        msg = f"Metadata missing required fields: {missing}"
        raise SchemaValidationError(msg)
