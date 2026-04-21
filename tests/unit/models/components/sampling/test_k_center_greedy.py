# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for k-center-greedy coreset selection."""

from unittest.mock import patch

import pytest
import torch

from anomalib.models.components.sampling.k_center_greedy import KCenterGreedy


class TestKCenterGreedy:
    """Tests for the KCenterGreedy coreset sampler."""

    @staticmethod
    def test_coreset_size() -> None:
        """Coreset should have exactly the expected number of elements."""
        embedding = torch.randn(100, 16)
        sampler = KCenterGreedy(embedding=embedding, sampling_ratio=0.1)
        idxs = sampler.select_coreset_idxs()
        assert len(idxs) == 10

    @staticmethod
    def test_initial_point_included() -> None:
        """The random starting point must be included in the coreset (gh-3459).

        Mocks ``torch.randint`` so the initial index is deterministic and then
        verifies that the exact index appears as the first coreset element.
        """
        embedding = torch.randn(100, 16)
        sampler = KCenterGreedy(embedding=embedding, sampling_ratio=0.1)

        initial_idx = 7
        fake_randint = lambda *_args, **_kwargs: torch.tensor(initial_idx)  # noqa: E731

        with patch("torch.randint", side_effect=fake_randint):
            idxs = sampler.select_coreset_idxs()

        assert idxs[0] == initial_idx, (
            f"First coreset element should be the initial random point {initial_idx}, got {idxs[0]}"
        )
        assert len(idxs) == 10

    @staticmethod
    def test_no_duplicate_indices() -> None:
        """Coreset indices should all be unique."""
        embedding = torch.randn(200, 32)
        sampler = KCenterGreedy(embedding=embedding, sampling_ratio=0.1)
        idxs = sampler.select_coreset_idxs()
        assert len(idxs) == len(set(idxs))

    @staticmethod
    def test_sample_coreset_shape() -> None:
        """sample_coreset should return a tensor with the correct shape."""
        embedding = torch.randn(200, 64)
        sampler = KCenterGreedy(embedding=embedding, sampling_ratio=0.05)
        coreset = sampler.sample_coreset()
        assert coreset.shape == (10, 64)

    @staticmethod
    def test_coreset_size_one() -> None:
        """Edge case: coreset_size == 1 should return only the initial point."""
        embedding = torch.randn(50, 8)
        sampler = KCenterGreedy(embedding=embedding, sampling_ratio=0.02)
        assert sampler.coreset_size == 1

        initial_idx = 23
        fake_randint = lambda *_args, **_kwargs: torch.tensor(initial_idx)  # noqa: E731

        with patch("torch.randint", side_effect=fake_randint):
            idxs = sampler.select_coreset_idxs()

        assert len(idxs) == 1
        assert idxs[0] == initial_idx

    @staticmethod
    def test_indices_within_bounds() -> None:
        """All returned indices should be valid embedding row indices."""
        n = 150
        embedding = torch.randn(n, 16)
        sampler = KCenterGreedy(embedding=embedding, sampling_ratio=0.1)
        idxs = sampler.select_coreset_idxs()
        assert all(0 <= i < n for i in idxs)

    @staticmethod
    def test_coreset_size_zero_raises() -> None:
        """Construction must raise ValueError when sampling_ratio yields coreset_size <= 0.

        A very small ``sampling_ratio`` combined with a small embedding set
        can yield ``coreset_size == 0``. The class should fail loudly at
        construction time rather than silently returning an empty or incorrect
        coreset later.
        """
        with pytest.raises(ValueError, match="coreset_size must be a positive integer"):
            KCenterGreedy(torch.randn(10, 4), sampling_ratio=0.01)
