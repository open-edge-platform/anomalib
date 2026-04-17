# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``ExportableCenterCrop`` size handling."""

import pytest
import torch

from anomalib.data.transforms.center_crop import ExportableCenterCrop


class TestExportableCenterCropSize:
    """Regression tests for the ``__init__`` size parsing."""

    @staticmethod
    def test_int_size_is_squared() -> None:
        """A bare ``int`` must become ``[size, size]``."""
        assert ExportableCenterCrop(size=224).size == [224, 224]

    @staticmethod
    def test_sequence_size_is_copied() -> None:
        """A list/tuple of ints must be preserved as a list."""
        assert ExportableCenterCrop(size=(224, 256)).size == [224, 256]
        assert ExportableCenterCrop(size=[320, 320]).size == [320, 320]

    @staticmethod
    def test_str_size_is_rejected() -> None:
        """A ``str`` must raise rather than be split into characters.

        Regression for PR #3539: ``str`` is a ``Sequence``, so the previous
        ``isinstance(size, Sequence)`` check silently turned ``"224"`` into
        the 3-element list ``["2", "2", "4"]``.
        """
        with pytest.raises(TypeError, match="size must be int or Sequence"):
            ExportableCenterCrop(size="224")

    @staticmethod
    def test_bytes_size_is_rejected() -> None:
        """Byte sequences must also raise rather than expand element-wise."""
        with pytest.raises(TypeError):
            ExportableCenterCrop(size=b"224")

    @staticmethod
    def test_transform_still_works_for_int_size() -> None:
        """The downstream transform must succeed with an ``int`` size."""
        transform = ExportableCenterCrop(size=224)
        image = torch.randn(3, 256, 256)
        output = transform(image)
        assert output.shape == (3, 224, 224)
