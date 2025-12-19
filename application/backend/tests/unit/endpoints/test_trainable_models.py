# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from api.endpoints.trainable_models_endpoints import _load_model_metadata  # noqa: PLC2701


def test_list_trainable_models_recommended_first(fxt_client):
    """Test that recommended models appear first in the list."""
    _load_model_metadata.cache_clear()

    response = fxt_client.get("/api/trainable-models")
    body = response.json()

    models = body["trainable_models"]
    recommended_models = [m for m in models if m["recommended"]]
    non_recommended_models = [m for m in models if not m["recommended"]]

    # If there are both recommended and non-recommended models,
    # recommended should come first
    if recommended_models and non_recommended_models:
        first_non_recommended_idx = next(i for i, m in enumerate(models) if not m["recommended"])
        last_recommended_idx = max(i for i, m in enumerate(models) if m["recommended"])
        assert last_recommended_idx < first_non_recommended_idx
