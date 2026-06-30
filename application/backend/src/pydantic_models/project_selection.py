# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import StrEnum

from pydantic import BaseModel

from utils.short_uuid import ShortUUID


class StartupProjectSelectionSource(StrEnum):
    LAST_USED = "last_used"
    ACTIVE_PIPELINE = "active_pipeline"
    FIRST_PROJECT = "first_project"
    NONE = "none"


class StartupProjectSelection(BaseModel):
    project_id: ShortUUID | None = None
    source: StartupProjectSelectionSource


class LastUsedProjectUpdate(BaseModel):
    project_id: ShortUUID
