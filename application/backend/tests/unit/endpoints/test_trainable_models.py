# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

from fastapi import status

from api.endpoints.trainable_models_endpoints import _get_trainable_models  # noqa: PLC2701


def test_list_trainable_models(fxt_client):
    _get_trainable_models.cache_clear()
    # Mock anomalib.models.list_models to return a predictable set
    with patch("api.endpoints.trainable_models_endpoints.list_models", return_value={"padim", "patchcore"}):
        response = fxt_client.get("/api/trainable-models")

    assert response.status_code == status.HTTP_200_OK
    body = response.json()
    assert body == {"trainable_models": ["padim", "patchcore"]} or body == {"trainable_models": ["patchcore", "padim"]}
