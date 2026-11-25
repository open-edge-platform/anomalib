# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for connected component labeling."""

from collections.abc import Callable

import pytest
import torch

from anomalib.utils.cv import connected_components_cpu, connected_components_gpu

LABELERS = [connected_components_cpu, connected_components_gpu]


class TestConnectedComponents:
    """Both labelers label regions 1..N and keep 0 for the background."""

    @staticmethod
    @pytest.mark.parametrize("labeler", LABELERS)
    def test_fully_anomalous_mask_is_one_region(labeler: Callable[[torch.Tensor], torch.Tensor]) -> None:
        """A mask without any background pixel is a single region labeled 1."""
        labels = labeler(torch.ones(1, 1, 4, 4))
        assert torch.all(labels == 1)

    @staticmethod
    @pytest.mark.parametrize("labeler", LABELERS)
    def test_fully_anomalous_batch_gets_one_label_per_image(
        labeler: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        """Two masks without background get two different non-zero labels."""
        labels = labeler(torch.ones(2, 1, 4, 4))
        assert torch.all(labels > 0)
        assert labels[0].unique().numel() == 1
        assert labels[1].unique().numel() == 1
        assert labels[0].unique() != labels[1].unique()

    @staticmethod
    @pytest.mark.parametrize("labeler", LABELERS)
    def test_background_stays_zero_and_labels_are_contiguous(
        labeler: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        """A batch mixing separate regions and a fully anomalous mask is labeled 0 for background and 1..N."""
        mask = torch.zeros(2, 1, 6, 6)
        mask[0, 0, 0:2, 0:2] = 1
        mask[0, 0, 4:6, 4:6] = 1
        mask[1, 0] = 1
        labels = labeler(mask)
        assert torch.all(labels[mask == 0] == 0)
        assert torch.all(labels[mask == 1] > 0)
        assert labels.unique().tolist() == [0, 1, 2, 3]
