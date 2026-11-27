# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SparseRandomProjection reproducibility and matrix generation."""

import pytest
import torch

from anomalib.models.components.dimensionality_reduction import SparseRandomProjection
from anomalib.models.components.sampling.k_center_greedy import KCenterGreedy


@pytest.mark.parametrize("n_features", [1, 64])
def test_sparse_random_projection_reproducible_with_random_state(n_features: int) -> None:
    """Setting random_state produces identical matrices regardless of global RNG state."""
    embedding = torch.randn(200, n_features)

    torch.manual_seed(0)
    proj1 = SparseRandomProjection(eps=0.9, random_state=42).fit(embedding)

    # Advance global RNG to verify random_state is independent of it.
    torch.manual_seed(999)
    _ = torch.randn(100)
    proj2 = SparseRandomProjection(eps=0.9, random_state=42).fit(embedding)

    assert torch.equal(proj1.sparse_random_matrix, proj2.sparse_random_matrix)


def test_sparse_random_projection_random_state_rows_are_independent() -> None:
    """Fixed random_state should not reuse the same column subset across rows."""
    embedding = torch.randn(200, 64)
    proj = SparseRandomProjection(eps=0.9, random_state=42).fit(embedding)
    matrix = proj.sparse_random_matrix

    unique_patterns = {
        tuple(torch.nonzero(matrix[row_idx], as_tuple=True)[0].tolist()) for row_idx in range(matrix.shape[0])
    }
    assert len(unique_patterns) > matrix.shape[0] // 2


def test_sparse_random_projection_random_state_preserves_global_rng() -> None:
    """Fitting with an explicit random_state must not mutate the global PyTorch RNG."""
    embedding = torch.randn(200, 64)
    torch.manual_seed(123)
    state_before = torch.get_rng_state()

    SparseRandomProjection(eps=0.9, random_state=42).fit(embedding)

    assert torch.equal(state_before, torch.get_rng_state())


def test_sparse_random_projection_and_k_center_greedy_reproducible_with_torch_seed() -> None:
    """With random_state=None, torch.manual_seed controls both projection and coreset selection."""
    embedding = torch.randn(200, 64)

    torch.manual_seed(42)
    proj1 = SparseRandomProjection(eps=0.9).fit(embedding)
    torch.manual_seed(42)
    proj2 = SparseRandomProjection(eps=0.9).fit(embedding)
    assert torch.equal(proj1.sparse_random_matrix, proj2.sparse_random_matrix)

    torch.manual_seed(42)
    idxs1 = KCenterGreedy(embedding=embedding, sampling_ratio=0.1).select_coreset_idxs()
    torch.manual_seed(42)
    idxs2 = KCenterGreedy(embedding=embedding, sampling_ratio=0.1).select_coreset_idxs()
    assert idxs1 == idxs2


def test_sparse_random_projection_different_random_states_differ() -> None:
    """Different random_state values should produce different projection matrices."""
    embedding = torch.randn(200, 64)
    proj1 = SparseRandomProjection(eps=0.9, random_state=42).fit(embedding)
    proj2 = SparseRandomProjection(eps=0.9, random_state=43).fit(embedding)

    assert not torch.equal(proj1.sparse_random_matrix, proj2.sparse_random_matrix)
