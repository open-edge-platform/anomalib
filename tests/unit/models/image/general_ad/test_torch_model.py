# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the GeneralAD model."""

import torch

from anomalib.models import GeneralAD, get_model


class TestGeneralAD:
    """Test the offline GeneralAD integration."""

    @staticmethod
    def test_get_model_and_forward() -> None:
        """GeneralAD should instantiate offline and produce anomaly outputs."""
        model = get_model("general_ad", pre_trained=False)
        assert isinstance(model, GeneralAD)

        model.eval()
        with torch.no_grad():
            output = model.model(torch.randn(2, 3, 256, 256))

        assert output.pred_score is not None
        assert output.anomaly_map is not None
        assert output.pred_score.shape == (2,)
        assert output.anomaly_map.shape == (2, 1, 256, 256)

    @staticmethod
    def test_compute_loss_is_finite() -> None:
        """GeneralAD training loss should be finite for a random batch."""
        model = GeneralAD(pre_trained=False)
        model.train()

        loss = model.model.compute_loss(torch.randn(2, 3, 256, 256))

        assert torch.isfinite(loss)
