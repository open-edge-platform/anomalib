# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi import status

from api.dependencies import get_model_service
from main import app
from pydantic_models import Model, ModelList
from services import ModelService


@pytest.fixture
def fxt_model(fxt_project):
    return Model(
        id=uuid4(),
        name="test_model",
        project_id=fxt_project.id,
        export_path="/path/to/model",
    )


@pytest.fixture
def fxt_model_service() -> MagicMock:
    model_service = MagicMock(spec=ModelService)
    app.dependency_overrides[get_model_service] = lambda: model_service
    return model_service


def test_get_models_empty(fxt_client, fxt_model_service, fxt_project):
    project_id = fxt_project.id
    fxt_model_service.get_model_list.return_value = ModelList(models=[])

    response = fxt_client.get(f"/api/projects/{project_id}/models")
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"models": []}
    fxt_model_service.get_model_list.assert_called_once_with(project_id=project_id)


def test_get_models(fxt_client, fxt_model_service, fxt_model, fxt_project):
    project_id = fxt_project.id
    fxt_model_service.get_model_list.return_value = ModelList(models=[fxt_model])

    response = fxt_client.get(f"/api/projects/{project_id}/models")
    assert response.status_code == status.HTTP_200_OK
    assert len(response.json()["models"]) == 1
    fxt_model_service.get_model_list.assert_called_once_with(project_id=project_id)
