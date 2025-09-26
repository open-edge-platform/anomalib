# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from repositories import MediaRepository
from repositories.binary_repo import ImageBinaryRepository
from services import MediaService


@pytest.fixture
def fxt_media_repository():
    """Fixture for a mock media repository."""
    return MagicMock(spec=MediaRepository)


@pytest.fixture
def fxt_image_binary_repo():
    """Fixture for a mock image binary repository."""
    return MagicMock(spec=ImageBinaryRepository)


@pytest.fixture
def fxt_media_service():
    """Fixture for MediaService - all methods are static."""
    return MediaService


@pytest.fixture(autouse=True)
def mock_db_context():
    """Mock the database context for all tests."""
    with patch("services.media_service.get_async_db_session_ctx") as mock_db_ctx:
        mock_session = AsyncMock()
        mock_db_ctx.return_value.__aenter__.return_value = mock_session
        mock_db_ctx.return_value.__aexit__.return_value = None
        yield mock_db_ctx


class TestMediaService:
    def test_get_media_list(self, fxt_media_service, fxt_media_repository, fxt_media_list, fxt_project):
        """Test getting media list."""
        fxt_media_repository.get_all.return_value = fxt_media_list.media

        with patch("services.media_service.MediaRepository") as mock_repo_class:
            mock_repo_class.return_value = fxt_media_repository

            result = asyncio.run(fxt_media_service.get_media_list(fxt_project.id))

        assert result == fxt_media_list
        fxt_media_repository.get_all.assert_called_once()

    def test_get_media_by_id(self, fxt_media_service, fxt_media_repository, fxt_media, fxt_project):
        """Test getting media by ID."""
        fxt_media_repository.get_by_id.return_value = fxt_media

        with patch("services.media_service.MediaRepository") as mock_repo_class:
            mock_repo_class.return_value = fxt_media_repository

            result = asyncio.run(fxt_media_service.get_media_by_id(fxt_project.id, fxt_media.id))

        assert result == fxt_media
        fxt_media_repository.get_by_id.assert_called_once_with(fxt_media.id)

    def test_get_media_by_id_not_found(self, fxt_media_service, fxt_media_repository, fxt_project):
        """Test getting media by ID when not found."""
        fxt_media_repository.get_by_id.return_value = None

        with patch("services.media_service.MediaRepository") as mock_repo_class:
            mock_repo_class.return_value = fxt_media_repository

            result = asyncio.run(fxt_media_service.get_media_by_id(fxt_project.id, "non-existent-id"))

        assert result is None
        fxt_media_repository.get_by_id.assert_called_once_with("non-existent-id")

    def test_get_media_file_path_success(self, fxt_media_service, fxt_media, fxt_project):
        """Test getting media file path successfully."""
        with patch("services.media_service.ImageBinaryRepository") as mock_bin_repo_class:
            mock_bin_repo = MagicMock()
            mock_bin_repo_class.return_value = mock_bin_repo
            mock_bin_repo.get_full_path.return_value = "/path/to/file.jpg"

            # Mock the get_media_by_id method
            fxt_media_service.get_media_by_id = AsyncMock(return_value=fxt_media)

            result = asyncio.run(fxt_media_service.get_media_file_path(fxt_project.id, fxt_media.id))

            assert result == "/path/to/file.jpg"
            mock_bin_repo.get_full_path.assert_called_once_with(filename=fxt_media.filename)

    def test_get_media_file_path_not_found(self, fxt_media_service, fxt_project):
        """Test getting media file path when media not found."""
        fxt_media_service.get_media_by_id = AsyncMock(return_value=None)

        with pytest.raises(FileNotFoundError) as exc_info:
            asyncio.run(fxt_media_service.get_media_file_path(fxt_project.id, "non-existent-id"))

        assert "Media with ID non-existent-id not found" in str(exc_info.value)

    def test_upload_image_success(self, fxt_media_service, fxt_upload_file, fxt_project):
        """Test successful image upload."""
        with patch("services.media_service.ImageBinaryRepository") as mock_bin_repo_class:
            mock_bin_repo = MagicMock()
            mock_bin_repo_class.return_value = mock_bin_repo
            mock_bin_repo.save_file = AsyncMock(return_value="/path/to/saved/file.jpg")

            # Mock the repository method
            mock_media_repo = MagicMock()
            mock_media_repo.save = AsyncMock(return_value=MagicMock())

            with patch("services.media_service.MediaRepository") as mock_repo_class:
                mock_repo_class.return_value = mock_media_repo

                result = asyncio.run(fxt_media_service.upload_image(fxt_project.id, fxt_upload_file, False))

            assert result is not None
            mock_bin_repo.save_file.assert_called_once()
            mock_media_repo.save.assert_called_once()

    def test_upload_image_rollback_on_error(self, fxt_media_service, fxt_upload_file, fxt_project):
        """Test image upload rollback on error."""
        with patch("services.media_service.ImageBinaryRepository") as mock_bin_repo_class:
            mock_bin_repo = MagicMock()
            mock_bin_repo_class.return_value = mock_bin_repo
            mock_bin_repo.save_file = AsyncMock(return_value="/path/to/saved/file.jpg")
            mock_bin_repo.delete_file = AsyncMock()

            # Mock the repository method to raise an error
            mock_media_repo = MagicMock()
            mock_media_repo.save = AsyncMock(side_effect=Exception("Database error"))

            with patch("services.media_service.MediaRepository") as mock_repo_class:
                mock_repo_class.return_value = mock_media_repo

                with pytest.raises(Exception, match="Database error"):
                    asyncio.run(fxt_media_service.upload_image(fxt_project.id, fxt_upload_file, False))

            # Verify rollback was attempted
            mock_bin_repo.delete_file.assert_called_once()

    def test_upload_image_rollback_file_not_found(self, fxt_media_service, fxt_upload_file, fxt_project):
        """Test image upload rollback when file deletion fails with FileNotFoundError."""
        with patch("services.media_service.ImageBinaryRepository") as mock_bin_repo_class:
            mock_bin_repo = MagicMock()
            mock_bin_repo_class.return_value = mock_bin_repo
            mock_bin_repo.save_file = AsyncMock(return_value="/path/to/saved/file.jpg")
            mock_bin_repo.delete_file = AsyncMock(side_effect=FileNotFoundError())

            # Mock the repository method to raise an error
            mock_media_repo = MagicMock()
            mock_media_repo.save = AsyncMock(side_effect=Exception("Database error"))

            with patch("services.media_service.MediaRepository") as mock_repo_class:
                mock_repo_class.return_value = mock_media_repo

                with pytest.raises(Exception, match="Database error"):
                    asyncio.run(fxt_media_service.upload_image(fxt_project.id, fxt_upload_file, False))

            # Verify rollback was attempted
            mock_bin_repo.delete_file.assert_called_once()

    def test_delete_media_success(self, fxt_media_service, fxt_media, fxt_project):
        """Test successful media deletion."""
        with patch("services.media_service.ImageBinaryRepository") as mock_bin_repo_class:
            mock_bin_repo = MagicMock()
            mock_bin_repo_class.return_value = mock_bin_repo
            mock_bin_repo.delete_file = AsyncMock()

            # Mock the repository method
            mock_media_repo = MagicMock()
            mock_media_repo.get_by_id = AsyncMock(return_value=fxt_media)
            mock_media_repo.delete_by_id = AsyncMock()

            with patch("services.media_service.MediaRepository") as mock_repo_class:
                mock_repo_class.return_value = mock_media_repo

                asyncio.run(fxt_media_service.delete_media(fxt_media.id, fxt_project.id))

            mock_bin_repo.delete_file.assert_called_once_with(filename=fxt_media.filename)
            mock_media_repo.delete_by_id.assert_called_once_with(fxt_media.id)

    def test_delete_media_not_found(self, fxt_media_service, fxt_project):
        """Test media deletion when media not found."""
        # Mock the repository method
        mock_media_repo = MagicMock()
        mock_media_repo.get_by_id = AsyncMock(return_value=None)

        with patch("services.media_service.MediaRepository") as mock_repo_class:
            mock_repo_class.return_value = mock_media_repo

            # Should not raise an exception, just return
            asyncio.run(fxt_media_service.delete_media("non-existent-id", fxt_project.id))

        mock_media_repo.get_by_id.assert_called_once_with("non-existent-id")

    def test_delete_media_file_not_found(self, fxt_media_service, fxt_media, fxt_project):
        """Test media deletion when file not found."""
        with patch("services.media_service.ImageBinaryRepository") as mock_bin_repo_class:
            mock_bin_repo = MagicMock()
            mock_bin_repo_class.return_value = mock_bin_repo
            mock_bin_repo.delete_file = AsyncMock(side_effect=FileNotFoundError())

            # Mock the repository method
            mock_media_repo = MagicMock()
            mock_media_repo.get_by_id = AsyncMock(return_value=fxt_media)
            mock_media_repo.delete_by_id = AsyncMock()

            with patch("services.media_service.MediaRepository") as mock_repo_class:
                mock_repo_class.return_value = mock_media_repo

                # Should not raise an exception, just log warning
                # Note: delete_by_id is not called when FileNotFoundError occurs
                asyncio.run(fxt_media_service.delete_media(fxt_media.id, fxt_project.id))

            mock_bin_repo.delete_file.assert_called_once_with(filename=fxt_media.filename)
            # delete_by_id should NOT be called when FileNotFoundError occurs
            mock_media_repo.delete_by_id.assert_not_called()

    def test_delete_media_other_error(self, fxt_media_service, fxt_media, fxt_project):
        """Test media deletion with other error."""
        with patch("services.media_service.ImageBinaryRepository") as mock_bin_repo_class:
            mock_bin_repo = MagicMock()
            mock_bin_repo_class.return_value = mock_bin_repo
            mock_bin_repo.delete_file = AsyncMock(side_effect=Exception("File system error"))

            # Mock the repository method
            mock_media_repo = MagicMock()
            mock_media_repo.get_by_id = AsyncMock(return_value=fxt_media)

            with patch("services.media_service.MediaRepository") as mock_repo_class:
                mock_repo_class.return_value = mock_media_repo

                with pytest.raises(Exception, match="File system error"):
                    asyncio.run(fxt_media_service.delete_media(fxt_media.id, fxt_project.id))
