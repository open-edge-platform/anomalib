# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import uuid

import pytest
from models import Job, JobType, Project


@pytest.fixture
def fxt_project():
    return Project(
        id=uuid.uuid4(),
        name="Test Project",
    )


@pytest.fixture
def fxt_job(fxt_project):
    return Job(
        project_id=fxt_project.id,
        type=JobType.TRAINING,
        payload={"model_name": "padim"},
        message="Test job created"
    )
