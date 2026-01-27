# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""This script generates an OpenAPI JSON file for the FastAPI application."""

import json
from pathlib import Path

from fastapi.openapi.utils import get_openapi

from main import app


def create_openapi(target_path: str) -> None:
    """Entry point for creating the OpenAPI specification"""
    # Ensure target directory exists
    output_path = Path(target_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        open_api = get_openapi(
            title=app.title,
            version=app.version,
            summary=app.summary,
            description=app.description,
            openapi_version=app.openapi_version,
            routes=app.routes,
        )
        json.dump(obj=open_api, fp=file, indent=2)
