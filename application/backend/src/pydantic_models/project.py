# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime

from pydantic import BaseModel, Field

from pydantic_models.base import BaseIDNameModel, Pagination


class Project(BaseIDNameModel):
    created_at: datetime | None = Field(default=None, description="Project creation timestamp", exclude=True)


class ProjectUpdate(BaseModel):
    name: str | None = None


class ProjectList(BaseModel):
    projects: list[Project]
    pagination: Pagination
