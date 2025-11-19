# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from geti_inspect.db.schema import MediaDB
from geti_inspect.pydantic_models import Media
from geti_inspect.repositories.mappers.base_mapper_interface import IBaseMapper


class MediaMapper(IBaseMapper):
    @staticmethod
    def to_schema(media: Media) -> MediaDB:
        return MediaDB(**media.model_dump(mode="json"))

    @staticmethod
    def from_schema(media_db: MediaDB) -> Media:
        return Media.model_validate(media_db, from_attributes=True)
