# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the GaussianKDE component."""

import pytest
import torch
from torch.distributions import MultivariateNormal

from anomalib.models.components.stats import GaussianKDE


@pytest.fixture
def offset_data() -> torch.Tensor:
    """Samples whose mean is far from the origin."""
    generator = torch.Generator().manual_seed(0)
    return torch.randn(200, 3, generator=generator) + torch.tensor([4.0, -2.0, 1.0])


def _reference_density(dataset: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
    """Gaussian KDE with Scott's rule: kernel covariance is ``factor**2`` times the data covariance."""
    num_samples, dimension = dataset.shape
    factor = num_samples ** (-1 / (dimension + 4))
    kernel_covariance = factor**2 * torch.cov(dataset.T)
    densities = [
        MultivariateNormal(point, covariance_matrix=kernel_covariance).log_prob(dataset).exp().mean()
        for point in queries
    ]
    return torch.stack(densities)


def test_density_matches_reference_for_off_center_data(offset_data: torch.Tensor) -> None:
    """The estimate equals the textbook KDE even when the data mean is not zero."""
    queries = offset_data[:5] + 0.1
    kde = GaussianKDE()
    kde.fit(offset_data.clone())

    assert torch.allclose(kde(queries), _reference_density(offset_data, queries), rtol=1e-3)


def test_fit_does_not_modify_the_dataset(offset_data: torch.Tensor) -> None:
    """Fitting leaves the caller's tensor untouched."""
    dataset = offset_data.clone()

    GaussianKDE().fit(dataset)

    assert torch.equal(dataset, offset_data)


def test_dataset_passed_to_the_constructor_is_fitted(offset_data: torch.Tensor) -> None:
    """``GaussianKDE(dataset=...)`` yields a fitted model equal to calling ``fit`` afterwards."""
    queries = offset_data[:5] + 0.1
    fitted_later = GaussianKDE()
    fitted_later.fit(offset_data.clone())

    fitted_in_constructor = GaussianKDE(dataset=offset_data.clone())

    assert torch.allclose(fitted_in_constructor(queries), fitted_later(queries))
