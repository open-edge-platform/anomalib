# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Metadata module for exported anomalib models.

Provides load/dump helpers and schema migration for ``metadata.json`` sidecar
files produced during model export.
"""

from anomalib.deploy.metadata._core import SchemaValidationError, dump_metadata, load_metadata
from anomalib.deploy.metadata.migration import CURRENT_SCHEMA_VERSION

__all__ = ["CURRENT_SCHEMA_VERSION", "SchemaValidationError", "dump_metadata", "load_metadata"]
