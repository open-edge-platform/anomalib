# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for safe pre-processing transform specifications."""

import pytest
from torchvision.transforms import v2

from anomalib.data.transforms import ExportableCenterCrop
from anomalib.pre_processing.utils.spec import spec_to_transform, transform_to_spec


@pytest.mark.parametrize(
    "transform",
    [
        v2.Compose([
            v2.Resize((128, 192), interpolation=v2.InterpolationMode.BICUBIC, antialias=False),
            v2.Normalize([0.1, 0.2, 0.3], [0.4, 0.5, 0.6]),
        ]),
        v2.CenterCrop((96, 128)),
        v2.Grayscale(num_output_channels=3),
        ExportableCenterCrop((64, 80)),
    ],
)
def test_transform_round_trip(transform: v2.Transform) -> None:
    """Transform specs reconstruct equivalent transforms."""
    spec = transform_to_spec(transform)
    reconstructed = spec_to_transform(spec)

    assert transform_to_spec(reconstructed) == spec


def test_none_round_trip() -> None:
    """None remains None."""
    assert transform_to_spec(None) is None
    assert spec_to_transform(None) is None


def test_rejects_unsafe_class_path() -> None:
    """Deserializer rejects imports outside approved namespaces."""
    with pytest.raises(ValueError, match="Unsupported transform class path"):
        spec_to_transform({"class_path": "os.system", "init_args": {}})


def test_rejects_unsupported_transform() -> None:
    """Serializer rejects transforms outside approved namespaces."""

    class Unsupported(v2.Transform):
        """Test-only unsupported transform."""

        @staticmethod
        def transform(inpt: object, params: object) -> object:
            del params  # unused
            return inpt

    with pytest.raises(ValueError, match="Unsupported transform class"):
        transform_to_spec(Unsupported())
