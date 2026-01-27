# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from db.schema import SourceDB
from pydantic_models.source import Source, SourceAdapter, SourceType
from repositories.mappers.base_mapper_interface import IBaseMapper


class SourceMapper(IBaseMapper):
    """Mapper for Source model <-> Source schema conversions."""

    # Define fields to exclude from config_data (common fields)
    _COMMON_FIELDS: set[str] = {"id", "project_id", "name", "source_type", "created_at", "updated_at"}

    @staticmethod
    def from_schema(source_db: SourceDB) -> Source:
        """Convert Source schema to Source model."""

        config_data = source_db.config_data or {}
        return SourceAdapter.validate_python({
            "id": source_db.id,
            "project_id": source_db.project_id,
            "name": source_db.name,
            "source_type": SourceType(source_db.source_type),
            **config_data,
        })

    @staticmethod
    def to_schema(source: Source) -> SourceDB:
        """Convert Source model to Source schema."""
        if source is None:
            raise ValueError("Source config cannot be None")

        source_dict = SourceAdapter.dump_python(source, exclude=SourceMapper._COMMON_FIELDS, exclude_none=True)

        return SourceDB(
            id=str(source.id),
            project_id=str(source.project_id),
            name=source.name,
            source_type=source.source_type.value,
            config_data=source_dict,
        )
