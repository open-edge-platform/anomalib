# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel


class StartupProjectSelectionSource(StrEnum):
    LAST_USED = "last_used"
    ACTIVE_PIPELINE = "active_pipeline"
    FIRST_PROJECT = "first_project"
    NONE = "none"


class StartupProjectSelection(BaseModel):
    project_id: UUID | None = None
    source: StartupProjectSelectionSource


class LastUsedProjectUpdate(BaseModel):
    project_id: UUID
