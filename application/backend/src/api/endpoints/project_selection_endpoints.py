# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated

from fastapi import APIRouter, Body, Depends, HTTPException, status

from api.dependencies import get_project_selection_service
from api.endpoints import API_PREFIX
from pydantic_models import LastUsedProjectUpdate, StartupProjectSelection
from services import ProjectSelectionService, ResourceNotFoundError

router = APIRouter(prefix=f"{API_PREFIX}/projects", tags=["Project"])


@router.get("/startup-selection")
async def get_startup_project_selection(
    project_selection_service: Annotated[ProjectSelectionService, Depends(get_project_selection_service)],
) -> StartupProjectSelection:
    """Return the project that should be restored when the app starts."""
    return await project_selection_service.get_startup_project_selection()


@router.put(
    "/last-used",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        status.HTTP_204_NO_CONTENT: {"description": "Last used project stored successfully"},
        status.HTTP_404_NOT_FOUND: {"description": "Project not found"},
    },
)
async def update_last_used_project(
    project_selection: Annotated[LastUsedProjectUpdate, Body()],
    project_selection_service: Annotated[ProjectSelectionService, Depends(get_project_selection_service)],
) -> None:
    """Persist the project that should be restored on the next application start."""
    try:
        await project_selection_service.set_last_used_project(project_selection.project_id)
    except ResourceNotFoundError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=error.message) from error
