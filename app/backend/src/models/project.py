# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pydantic import Field
from datetime import datetime

from models.base import BaseIDNameModel
from pydantic import BaseModel


class Project(BaseIDNameModel):
    created_at: datetime | None = Field(default=None, description="Project creation timestamp", exclude=True)


class ProjectList(BaseModel):
    projects: list[Project]
