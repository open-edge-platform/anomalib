# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import asyncio
import os
from io import BytesIO
from uuid import UUID, uuid4

from fastapi import UploadFile
from loguru import logger
from PIL import Image

from db import get_async_db_session_ctx
from pydantic_models import Media, MediaList
from repositories import MediaRepository
from repositories.binary_repo import ImageBinaryRepository

THUMBNAIL_SIZE = 256  # Max width/height for thumbnails in pixels


class MediaService:
    @staticmethod
    def _get_thumbnail_filename(media_id: UUID) -> str:
        """Generate thumbnail filename for a given media ID."""
        return f"thumb_{media_id}.png"

    @staticmethod
    async def get_media_list(project_id: UUID) -> MediaList:
        async with get_async_db_session_ctx() as session:
            repo = MediaRepository(session, project_id=project_id)
            return MediaList(media=await repo.get_all())

    @staticmethod
    async def get_media_by_id(project_id: UUID, media_id: UUID) -> Media | None:
        async with get_async_db_session_ctx() as session:
            repo = MediaRepository(session, project_id=project_id)
            return await repo.get_by_id(media_id)

    @classmethod
    async def get_media_file_path(cls, project_id: UUID, media_id: UUID) -> str:
        media = await cls.get_media_by_id(project_id=project_id, media_id=media_id)
        if media is None:
            raise FileNotFoundError(f"Media with ID {media_id} not found.")
        bin_repo = ImageBinaryRepository(project_id=project_id)
        return bin_repo.get_full_path(filename=media.filename)

    @classmethod
    async def get_thumbnail_file_path(cls, project_id: UUID, media_id: UUID) -> str:
        """Get the file path for a media's thumbnail."""
        media = await cls.get_media_by_id(project_id=project_id, media_id=media_id)
        if media is None:
            raise FileNotFoundError(f"Media with ID {media_id} not found.")
        bin_repo = ImageBinaryRepository(project_id=project_id)
        thumbnail_filename = cls._get_thumbnail_filename(media_id)
        return bin_repo.get_full_path(filename=thumbnail_filename)

    @classmethod
    async def upload_image(cls, project_id: UUID, file: UploadFile, image_bytes: bytes, is_anomalous: bool) -> Media:
        # Generate unique filename and media ID
        media_id = uuid4()
        
        if file.filename is None or file.size is None:
            raise ValueError("File must have a filename and size")

        extension = list(os.path.splitext(file.filename)).pop().lower()
        filename = f"{media_id}{extension}"
        bin_repo = ImageBinaryRepository(project_id=project_id)
        saved_media: Media | None = None

        async with get_async_db_session_ctx() as session:
            media_repo = MediaRepository(session, project_id=project_id)
            try:
                # Save original file to filesystem
                saved_file_path = await bin_repo.save_file(
                    filename=filename,
                    content=image_bytes,
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
                # Attempt to delete the files if they were saved
                try:
                    await cls._delete_media_file(project_id=project_id, filename=filename)
                except Exception as delete_error:
                    logger.error(f"Failed to delete media file during rollback: {delete_error}")
                if saved_media is not None:
                    await media_repo.delete_by_id(saved_media.id)
                raise e
            return saved_media

    @classmethod
    async def delete_media(cls, media_id: UUID, project_id: UUID) -> None:
        async with get_async_db_session_ctx() as session:
            media_repo = MediaRepository(session, project_id=project_id)
            media = await media_repo.get_by_id(media_id)
            if media is None:
                logger.warning(f"Media with ID {media_id} not found for deletion.")
                return
            thumbnail_filename = cls._get_thumbnail_filename(media_id)
            await cls._delete_media_file(project_id=project_id, filename=media.filename)
            await cls._delete_media_file(project_id=project_id, filename=thumbnail_filename)

    @staticmethod
    async def _delete_media_file(project_id: UUID, filename: str) -> None:
        bin_repo = ImageBinaryRepository(project_id=project_id)

        try:
            await bin_repo.delete_file(filename=filename)
        except FileNotFoundError:
            logger.warning(f"File `{filename}` not found during media deletion.")
        except Exception as e:
            logger.error(f"Error deleting file `{filename}`: {e}")
            raise e

    @classmethod
    async def generate_thumbnail(
        cls,
        project_id: UUID,
        media_id: UUID,
        image_bytes: bytes,
        height_px: int = THUMBNAIL_SIZE,
        width_px: int = THUMBNAIL_SIZE,
    ) -> None:
        """Create and persist a PNG thumbnail for a media item.

        Resizes the input image to fit within width_px x height_px while
        preserving aspect ratio. Intended to run as a background task so the
        upload path stays fast.

        Args:
            project_id: Identifier of the owning project.
            media_id: Identifier of the media item the thumbnail belongs to.
            image_bytes: Original image bytes used to create the thumbnail.
            height_px: Maximum thumbnail height in pixels. Defaults to
                ``THUMBNAIL_SIZE``.
            width_px: Maximum thumbnail width in pixels. Defaults to
                ``THUMBNAIL_SIZE``.

        Returns:
            None

        Raises:
            Exception: If processing or persistence fails
        """

        def _create() -> bytes:
            with Image.open(BytesIO(image_bytes)) as img:
                # Preserve aspect ratio while fitting within the box
                img.thumbnail((width_px, height_px))
                with BytesIO() as buf:
                    img.save(buf, format="PNG")
                    return buf.getvalue()

        thumbnail_bytes = await asyncio.to_thread(_create)
        thumbnail_filename = cls._get_thumbnail_filename(media_id)
        bin_repo = ImageBinaryRepository(project_id=project_id)
        try:
            thumbnail_path = await bin_repo.save_file(
                filename=thumbnail_filename,
                content=thumbnail_bytes,
            )
            logger.info(f"Saved thumbnail: {thumbnail_path}")
        except Exception as e:
            logger.error(f"Rolling back media thumbnail generation due to error: {e}")
            # Attempt to delete the files if they were saved
            try:
                await cls._delete_media_file(project_id=project_id, filename=thumbnail_filename)
            except Exception as delete_error:
                logger.error(f"Failed to delete media file during rollback: {delete_error}")
            raise e
