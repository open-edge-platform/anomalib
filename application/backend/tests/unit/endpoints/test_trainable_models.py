# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

from fastapi import status

from pydantic_models import ModelFamily, ModelInfo, TrainableModelList, TrainingTime


def test_list_trainable_models(fxt_client):
    # Mock anomalib.models.list_models to return a predictable set
    mock_response = {
        "trainable_models": [
            {
                "name": "Padim",
                "class_name": "padim",
                "training_time": "coffee",
                "recommended": False,
                "license": "Apache-2.0",
                "model_family": ["memory_bank", "patch_based", "distribution_map"],
            },
        ]
    }
    with patch(
        "api.endpoints.trainable_models_endpoints._get_trainable_models",
        return_value=TrainableModelList(
            trainable_models=[
                ModelInfo(
                    name="Padim",
                    class_name="padim",
                    training_time=TrainingTime.COFFEE,
                    model_family=[ModelFamily.MEMORY_BANK, ModelFamily.PATCH_BASED, ModelFamily.DISTRIBUTION_MAP],
                )
            ]
        ),
    ):
        response = fxt_client.get("/api/trainable-models")

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == mock_response
