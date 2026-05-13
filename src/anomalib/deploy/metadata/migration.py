# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Schema migration for metadata.json files."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

CURRENT_SCHEMA_VERSION = "1.0"


def upgrade_to_latest(metadata: dict) -> dict:
    """Upgrade metadata dict to the current schema version.

    Applies each migration step in sequence. Pure function — does not mutate input.

    Args:
        metadata (dict): Raw metadata dictionary, possibly from an older schema.

    Returns:
        dict: Metadata upgraded to ``CURRENT_SCHEMA_VERSION``.
    """
    metadata = metadata.copy()
    version = metadata.get("schema_version")

    if version is None:
        metadata = _legacy_to_v1(metadata)

    if metadata["schema_version"] > CURRENT_SCHEMA_VERSION:
        logger.warning(
            "Metadata schema %s is newer than supported (%s). "
            "Unknown fields will be ignored. Consider upgrading anomalib.",
            metadata["schema_version"],
            CURRENT_SCHEMA_VERSION,
        )

    return metadata


def _legacy_to_v1(raw: dict) -> dict:
    """Convert legacy (pre-schema) metadata to schema 1.0.

    Args:
        raw (dict): Legacy metadata dictionary without ``schema_version``.

    Returns:
        dict: Schema 1.0 metadata with sensible defaults.
    """
    return {
        "schema_version": "1.0",
        "anomalib_version": "unknown",
        "model": raw.get("model", "unknown"),
        "preprocess": [],
        "postprocess": {
            "image_sensitivity": 0.5,
            "pixel_sensitivity": 0.5,
        },
    }
