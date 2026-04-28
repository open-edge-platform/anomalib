# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for SquarePad transform."""

import torch

from anomalib.data.transforms.square_pad import SquarePad


class TestSquarePadFloat:
    """Tests for ``SquarePad`` with float (image) tensors."""

    @staticmethod
    def test_landscape_becomes_square() -> None:
        """Landscape image is padded along height to become square."""
        transform = SquarePad()
        image = torch.randn(3, 200, 300)
        result = transform(image)
        assert result.shape == torch.Size([3, 300, 300])

    @staticmethod
    def test_portrait_becomes_square() -> None:
        """Portrait image is padded along width to become square."""
        transform = SquarePad()
        image = torch.randn(3, 400, 250)
        result = transform(image)
        assert result.shape == torch.Size([3, 400, 400])

    @staticmethod
    def test_already_square_is_unchanged() -> None:
        """Square input is returned without modification."""
        transform = SquarePad()
        image = torch.randn(3, 256, 256)
        result = transform(image)
        assert result.shape == torch.Size([3, 256, 256])
        torch.testing.assert_close(result, image)

    @staticmethod
    def test_uses_replicate_padding() -> None:
        """Float tensors are padded by replicating edge pixels, not zeros."""
        transform = SquarePad()
        image = torch.ones(1, 2, 4)
        result = transform(image)
        # Replicate fills with 1.0, so entire output should be 1.0.
        assert result.shape == torch.Size([1, 4, 4])
        torch.testing.assert_close(result, torch.ones(1, 4, 4))

    @staticmethod
    def test_batched_input() -> None:
        """Batched (4-D) tensors are handled correctly."""
        transform = SquarePad()
        batch = torch.randn(2, 3, 100, 200)
        result = transform(batch)
        assert result.shape == torch.Size([2, 3, 200, 200])


class TestSquarePadBool:
    """Tests for ``SquarePad`` with boolean (mask) tensors."""

    @staticmethod
    def test_mask_becomes_square() -> None:
        """Boolean mask is padded to a square shape."""
        transform = SquarePad()
        mask = torch.ones(1, 100, 200, dtype=torch.bool)
        result = transform(mask)
        assert result.shape == torch.Size([1, 200, 200])

    @staticmethod
    def test_mask_uses_constant_zero_padding() -> None:
        """Boolean masks are padded with False (constant 0), not replicate."""
        transform = SquarePad()
        mask = torch.ones(1, 2, 4, dtype=torch.bool)
        result = transform(mask)
        assert result.shape == torch.Size([1, 4, 4])
        # Original region should stay True.
        assert result[0, 1:3, :].all()
        # Padded rows should be False.
        assert not result[0, 0, :].any()
        assert not result[0, 3, :].any()

    @staticmethod
    def test_mask_dtype_preserved() -> None:
        """Output dtype remains bool for boolean inputs."""
        transform = SquarePad()
        mask = torch.zeros(1, 50, 80, dtype=torch.bool)
        result = transform(mask)
        assert result.dtype == torch.bool
