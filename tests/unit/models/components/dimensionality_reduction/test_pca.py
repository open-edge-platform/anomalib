# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the PCA component."""

import pytest
import torch

from anomalib.models.components import PCA


def _low_rank_data(rank: int, n_features: int = 10, noise: float = 1e-3) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    latent = torch.randn(200, rank, generator=generator)
    mixing = torch.randn(rank, n_features, generator=generator)
    return latent @ mixing + noise * torch.randn(200, n_features, generator=generator)


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_variance_ratio_keeps_every_needed_component(rank: int) -> None:
    """The number of kept components is the count needed to reach the variance ratio."""
    data = _low_rank_data(rank)
    singular_values = torch.linalg.svdvals(data - data.mean(dim=0)).double()
    ratios = torch.cumsum(singular_values**2, dim=0) / torch.sum(singular_values**2)
    expected = int((ratios < 0.99).sum().item()) + 1  # first count whose cumulative ratio reaches 0.99

    pca = PCA(n_components=0.99)
    pca.fit(data.clone())

    assert int(pca.num_components) == expected == rank
    assert pca.singular_vectors.shape == (data.shape[1], expected)
    assert pca.singular_values.shape == (expected,)


@pytest.fixture
def offset_data() -> torch.Tensor:
    """Data whose mean is far from the origin."""
    generator = torch.Generator().manual_seed(0)
    return torch.randn(50, 6, generator=generator) + 3.0


def test_fit_and_transform_do_not_modify_their_input(offset_data: torch.Tensor) -> None:
    """The tensors passed to ``fit``, ``transform`` and ``fit_transform`` are left untouched."""
    expected = offset_data.clone()
    pca = PCA(n_components=3)

    pca.fit(offset_data)
    assert torch.equal(offset_data, expected)

    pca.transform(offset_data)
    assert torch.equal(offset_data, expected)

    pca.fit_transform(offset_data)
    assert torch.equal(offset_data, expected)


def test_transformed_training_data_is_centered(offset_data: torch.Tensor) -> None:
    """Projecting the data the PCA was fitted on yields zero-mean components."""
    pca = PCA(n_components=3)
    pca.fit(offset_data)

    projected = pca.transform(offset_data)

    assert torch.allclose(projected.mean(dim=0), torch.zeros(3), atol=1e-5)


def test_inverse_transform_restores_the_mean(offset_data: torch.Tensor) -> None:
    """With every component kept, ``inverse_transform(transform(x))`` returns ``x``."""
    pca = PCA(n_components=offset_data.shape[1])
    pca.fit(offset_data.clone())

    reconstructed = pca.inverse_transform(pca.transform(offset_data.clone()))

    assert torch.allclose(reconstructed, offset_data, atol=1e-4)


def test_fit_transform_matches_fit_then_transform(offset_data: torch.Tensor) -> None:
    """``fit_transform`` accepts a variance ratio and agrees with ``fit`` followed by ``transform``."""
    fitted = PCA(n_components=0.9)
    fitted.fit(offset_data)

    pca = PCA(n_components=0.9)
    projected = pca.fit_transform(offset_data)

    assert projected.shape[1] == int(fitted.num_components)
    assert torch.allclose(projected, fitted.transform(offset_data), atol=1e-5)


def test_integer_n_components_keeps_that_many_columns(offset_data: torch.Tensor) -> None:
    """An integer ``n_components`` of 1 is a component count, not a variance ratio."""
    assert PCA(n_components=1).fit_transform(offset_data).shape[1] == 1


def test_fit_transform_keeps_float64_input(offset_data: torch.Tensor) -> None:
    """``fit_transform`` works on and returns float64 data."""
    assert PCA(n_components=3).fit_transform(offset_data.double()).dtype == torch.float64
