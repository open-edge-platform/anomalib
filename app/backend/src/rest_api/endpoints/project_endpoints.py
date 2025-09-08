# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Body, Depends, HTTPException, status

from models import Project, ProjectList
from rest_api.dependencies import get_project_id, get_project_service
from rest_api.endpoints import API_PREFIX
from services import ProjectService

logger = logging.getLogger(__name__)

project_api_prefix_url = API_PREFIX + "/projects"
project_router = APIRouter(
    prefix=project_api_prefix_url,
    tags=["Project"],
)


@project_router.get("")
async def get_projects(project_service: Annotated[ProjectService, Depends(get_project_service)]) -> ProjectList:
    """Endpoint to get list of all projects"""
    return await project_service.get_project_list()


@project_router.post("")
async def create_project(
    project_service: Annotated[ProjectService, Depends(get_project_service)],
    project: Annotated[Project, Body()],
) -> Project:
    """Endpoint to create a new project"""
    return await project_service.create_project(project)


@project_router.get("/{project_id}")
async def get_project_by_id(
    project_service: Annotated[ProjectService, Depends(get_project_service)],
    project_id: Annotated[UUID, Depends(get_project_id)],
) -> Project:
    """Endpoint to get project metadata by ID"""
    project = await project_service.get_project_by_id(project_id)
    if project is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    return project
