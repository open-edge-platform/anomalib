# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for MH-PatchCore statistical components."""

import numpy as np
import pytest
import torch
from sklearn.decomposition import IncrementalPCA

from anomalib.models.image.mh_patchcore.components import StreamingPCA


@pytest.mark.parametrize("variance_ratio", [0.0, -0.1, 1.1, True])
def test_variance_ratio_validation(variance_ratio: float) -> None:
    """Variance retention must be numeric and within the supported interval."""
    with pytest.raises(ValueError, match="variance_ratio must be in"):
        StreamingPCA(variance_ratio=variance_ratio)


def test_streaming_fit_preserves_undersized_boundary_batches() -> None:
    """Undersized first and final batches should contribute exactly once."""
    generator = torch.Generator().manual_seed(7)
    embeddings = torch.randn(12, 4, generator=generator)
    batches = list(embeddings.split([2, 4, 5, 1]))
    pca = StreamingPCA(variance_ratio=0.9)

    for batch in batches:
        pca.update(batch)
    pca.finalize()

    reference = IncrementalPCA(n_components=None)
    reference.partial_fit(torch.cat(batches[:2]).numpy())
    reference.partial_fit(torch.cat(batches[2:]).numpy())
    cumulative_variance = np.cumsum(reference.explained_variance_ratio_)
    expected_components = int(np.flatnonzero(cumulative_variance >= 0.9)[0] + 1)
    expected = reference.transform(embeddings.numpy())[:, :expected_components].astype(np.float32)

    assert pca.num_components.item() == expected_components
    assert pca.components.shape == (expected_components, embeddings.shape[1])
    assert pca.components.dtype == torch.float32
    assert pca.mean.shape == (embeddings.shape[1],)
    assert pca.mean.dtype == torch.float64
    assert pca.projected_mean.shape == (expected_components,)
    assert pca.projected_mean.dtype == torch.float32
    torch.testing.assert_close(pca.mean, embeddings.double().mean(dim=0))
    torch.testing.assert_close(pca(embeddings), torch.from_numpy(expected), rtol=1e-5, atol=1e-6)
    assert torch.isfinite(pca(embeddings)).all()


def test_finalize_all_undersized_stream() -> None:
    """A complete stream smaller than its feature dimension should still fit."""
    values = np.random.default_rng(2).normal(size=(3, 8)).astype(np.float32)
    embeddings = torch.from_numpy(values)
    pca = StreamingPCA(variance_ratio=1.0)
    pca.update(embeddings[:1])
    pca.update(embeddings[1:])

    pca.finalize()

    assert pca.num_components.item() == len(embeddings)
    assert pca.components.shape == (len(embeddings), embeddings.shape[1])
    assert pca(embeddings).shape == (len(embeddings), len(embeddings))
    assert torch.isfinite(pca(embeddings)).all()


def test_persistent_state_restores_torch_transform() -> None:
    """A state-dict roundtrip should not require the fitting estimator."""
    embeddings = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    pca = StreamingPCA(variance_ratio=0.95)
    pca.update(embeddings[:4])
    pca.update(embeddings[4:])
    pca.finalize()
    expected = pca(embeddings)

    restored = StreamingPCA(variance_ratio=0.95)
    restored.load_state_dict(pca.state_dict())

    assert restored.is_fitted
    assert restored._estimator is None  # noqa: SLF001
    torch.testing.assert_close(restored(embeddings), expected)


def test_fitted_state_errors() -> None:
    """Invalid fitting-state transitions should fail clearly."""
    pca = StreamingPCA()
    embeddings = torch.randn(6, 4)

    with pytest.raises(RuntimeError, match="finalized before transform"):
        pca.transform(embeddings)
    with pytest.raises(RuntimeError, match="received no embedding batches"):
        pca.finalize()

    pca.update(embeddings)
    pca.finalize()
    with pytest.raises(RuntimeError, match="cannot be updated after finalization"):
        pca.update(embeddings)
    with pytest.raises(RuntimeError, match="already finalized"):
        pca.finalize()
