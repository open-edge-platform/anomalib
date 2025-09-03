# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Body, HTTPException, status

from models import Project, ProjectList
from rest_api.dependencies import get_project_service, get_project_id
from rest_api.endpoints import API_PREFIX
from services import ProjectService

logger = logging.getLogger(__name__)

project_api_prefix_url = API_PREFIX + "/projects"
project_router = APIRouter(
    prefix=project_api_prefix_url,
    tags=["Project"],
)


@project_router.get("", response_model=ProjectList)
async def get_projects(project_service: Annotated[ProjectService, Depends(get_project_service)]):
    return await project_service.get_project_list()


@project_router.post("", response_model=Project)
async def create_project(
    project_service: Annotated[ProjectService, Depends(get_project_service)],
    project: Annotated[Project, Body()],
):
    return await project_service.create_project(project)


@project_router.get("/{project_id}", response_model=Project)
async def get_project_by_id(
    project_service: Annotated[ProjectService, Depends(get_project_service)],
    project_id: Annotated[UUID, Depends(get_project_id)],
):
    project = await project_service.get_project_by_id(project_id)
    if project is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    return project
