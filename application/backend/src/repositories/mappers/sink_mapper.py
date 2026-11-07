# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from db.schema import SinkDB
from pydantic_models.sink import Sink, SinkAdapter, SinkType
from repositories.mappers.base_mapper_interface import IBaseMapper


class SinkMapper(IBaseMapper):
    """Mapper for Sink model <-> Sink schema conversions."""

    # Define fields to exclude from config_data (common fields)
    _COMMON_FIELDS: set[str] = {
        "id",
        "project_id",
        "name",
        "sink_type",
        "output_formats",
        "rate_limit",
        "created_at",
        "updated_at",
    }

    @staticmethod
    def from_schema(sink_db: SinkDB) -> Sink:
        """Convert Sink DB schema to Sink pydantic model."""

        config_data = sink_db.config_data or {}
        return SinkAdapter.validate_python({
            "id": sink_db.id,
            "project_id": sink_db.project_id,
            "name": sink_db.name,
            "sink_type": SinkType(sink_db.sink_type),
            "output_formats": sink_db.output_formats,
            "rate_limit": sink_db.rate_limit,
            **config_data,
        })

    @staticmethod
    def to_schema(sink: Sink) -> SinkDB:
        """Convert Sink pydantic model to Sink DB schema."""

        if sink is None:
            raise ValueError("Sink config cannot be None")

        sink_dict = SinkAdapter.dump_python(sink, exclude=SinkMapper._COMMON_FIELDS, exclude_none=True)

        return SinkDB(
            id=str(sink.id),
            project_id=str(sink.project_id),
            name=sink.name,
            sink_type=sink.sink_type,
            output_formats=sink.output_formats,
            rate_limit=sink.rate_limit,
            config_data=sink_dict,
        )
