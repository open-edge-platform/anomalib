# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime

from pydantic import BaseModel, Field

from models.base import BaseIDNameModel


class Project(BaseIDNameModel):
    created_at: datetime | None = Field(default=None, description="Project creation timestamp", exclude=True)


class ProjectList(BaseModel):
    projects: list[Project]
