# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the DFM torch model."""

import torch

from anomalib.models.image.dfm.torch_model import DFMModel, SingleClassGaussian


def test_single_class_gaussian_fit_stores_the_feature_mean() -> None:
    """``fit`` takes data shaped ``(n_features, n_samples)`` and stores the mean over the samples."""
    dataset = torch.randn(5, 40) + 2.0

    gaussian = SingleClassGaussian()
    gaussian.fit(dataset)

    assert torch.allclose(gaussian.mean_vec, dataset.mean(dim=1))


def test_dfm_model_with_nll_score_can_be_fitted_and_scored() -> None:
    """A DFM model with ``score_type="nll"`` finishes fitting and returns finite image scores."""
    torch.manual_seed(0)
    model = DFMModel(backbone="resnet18", layer="layer3", pre_trained=False, n_comps=0.9, score_type="nll")

    model.train()
    for _ in range(4):
        model(torch.rand(8, 3, 64, 64))
    model.fit()

    model.eval()
    output = model(torch.rand(2, 3, 64, 64))

    assert output.pred_score.shape == (2,)
    assert torch.isfinite(output.pred_score).all()
