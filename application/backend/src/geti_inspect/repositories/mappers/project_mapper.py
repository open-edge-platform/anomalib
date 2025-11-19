# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from geti_inspect.db.schema import ProjectDB
from geti_inspect.pydantic_models import Project
from geti_inspect.repositories.mappers.base_mapper_interface import IBaseMapper


class ProjectMapper(IBaseMapper):
    @staticmethod
    def to_schema(project: Project) -> ProjectDB:
        return ProjectDB(**project.model_dump(mode="json"))

    @staticmethod
    def from_schema(model_db: ProjectDB) -> Project:
        return Project.model_validate(model_db, from_attributes=True)
