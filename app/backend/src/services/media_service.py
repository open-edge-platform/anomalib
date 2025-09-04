# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import logging
import os
from uuid import UUID, uuid4

from fastapi import UploadFile
from sqlalchemy.ext.asyncio.session import AsyncSession

from models import Media, MediaList
from repositories import MediaRepository
from repositories.binary_repo import FILETYPE, BinaryRepository

logger = logging.getLogger(__name__)


class MediaService:
    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session

    def repository(self, project_id: str | UUID) -> MediaRepository:
        return MediaRepository(self.db_session, project_id=project_id)

    async def get_media_list(self, project_id: UUID) -> MediaList:
        repo = self.repository(project_id=str(project_id))
        return MediaList(media=await repo.get_all())

    async def get_media_by_id(self, project_id: UUID, media_id: UUID) -> Media | None:
        repo = self.repository(project_id=project_id)
        return await repo.get_by_id(media_id)

    async def get_media_file_path(self, project_id: UUID, media_id: UUID) -> str:
        media = await self.get_media_by_id(project_id=project_id, media_id=media_id)
        if media is None:
            raise FileNotFoundError(f"Media with ID {media_id} not found.")
        bin_repo = BinaryRepository(project_id=project_id)
        return bin_repo.get_full_path(filename=media.filename, file_type=FILETYPE.IMAGES)

    async def upload_image(self, project_id: UUID, file: UploadFile, is_anomalous: bool) -> Media:
        # Generate unique filename and media ID
        media_id = uuid4()
        extension = list(os.path.splitext(file.filename)).pop().lower()
        filename = f"{media_id}{extension}"
        bin_repo = BinaryRepository(project_id=str(project_id))
        media_repo = self.repository(str(project_id))
        saved_media: Media | None = None
        image_bytes = await file.read()

        try:
            # Save file to filesystem
            saved_file_path = await bin_repo.save_file(
                filename=filename,
                content=image_bytes,
                file_type=FILETYPE.IMAGES,
            )
            logger.info(f"Saved media file: {saved_file_path}")

            # Create media record in database
            media = Media(
                id=media_id,
                project_id=project_id,
                filename=filename,
                size=file.size,
                is_anomalous=is_anomalous,
            )
            saved_media = await media_repo.save(media)
        except Exception as e:
            logger.error(f"Rolling back media upload due to error: {e}")
            # Attempt to delete the file if it was saved
            try:
                await bin_repo.delete_file(filename=filename, file_type=FILETYPE.IMAGES)
            except FileNotFoundError:
                pass
            except Exception as delete_error:
                logger.error(f"Failed to delete file during rollback: {delete_error}")
            if saved_media is not None:
                await media_repo.delete(saved_media.id)
            raise e
        return saved_media

    async def delete_media(self, media_id: UUID, project_id: UUID) -> None:
        media_repo = self.repository(str(project_id))
        media = await media_repo.get_by_id(media_id)
        if media is None:
            logger.warning(f"Media with ID {media_id} not found for deletion.")
            return
        bin_repo = BinaryRepository(project_id=str(project_id))
        try:
            await bin_repo.delete_file(filename=media.filename, file_type=FILETYPE.IMAGES)
            await media_repo.delete(media_id)
        except FileNotFoundError:
            logger.warning(f"File {media.filename} not found during media deletion.")
        except Exception as e:
            logger.error(f"Error deleting file {media.filename}: {e}")
            raise e
