# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel


class TrainableModelList(BaseModel):
    """List wrapper for returning multiple trainable models."""

    trainable_models: list[str]
