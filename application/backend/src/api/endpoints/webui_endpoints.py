# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from settings import get_settings

settings = get_settings()
webui_router = APIRouter(tags=["Webui"])


@webui_router.get("/", include_in_schema=False)
@webui_router.get("/{full_path:path}", include_in_schema=False)
async def get_webui(full_path: str = "") -> FileResponse:  # noqa: ARG001
    """Get the webui index.html file."""
    if settings.static_files_dir and not (file_path := Path(settings.static_files_dir) / "index.html").exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)
