# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi import status

from api.dependencies import get_media_service
from main import app
from models import Media, MediaList
from services import MediaService


@pytest.fixture
def fxt_media(fxt_project):
    return Media(
        id=uuid4(),
        project_id=fxt_project.id,
        filename="test_image.jpg",
        size=64,
        is_anomalous=False,
    )


@pytest.fixture
def fxt_media_service() -> MagicMock:
    media_service = MagicMock(spec=MediaService)
    app.dependency_overrides[get_media_service] = lambda: media_service
    return media_service


def test_get_media_list_empty(fxt_client, fxt_media_service):
    project_id = uuid4()
    fxt_media_service.get_media_list.return_value = MediaList(media=[])

    response = fxt_client.get(f"/api/projects/{project_id}/images")
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"media": []}
    fxt_media_service.get_media_list.assert_called_once_with(project_id=project_id)


def test_get_media_list(fxt_client, fxt_media_service, fxt_media):
    project_id = uuid4()
    fxt_media_service.get_media_list.return_value = MediaList(media=[fxt_media])

    response = fxt_client.get(f"/api/projects/{project_id}/images")
    assert response.status_code == status.HTTP_200_OK
    assert len(response.json()["media"]) == 1
    fxt_media_service.get_media_list.assert_called_once_with(project_id=project_id)
