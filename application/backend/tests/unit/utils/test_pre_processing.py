# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock

import pytest
import torch
from anomalib.models import Dinomaly
from anomalib.pre_processing import PreProcessor
from torchvision.transforms.v2 import CenterCrop, Compose, InterpolationMode, Normalize, Resize

from utils.pre_processing import without_center_crop

NORMALIZE = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class TestWithoutCenterCrop:
    def test_replaces_resize_and_center_crop_with_resize_to_crop_size(self) -> None:
        pre_processor = PreProcessor(
            transform=Compose([Resize((448, 448), interpolation=InterpolationMode.BICUBIC), CenterCrop(392), NORMALIZE])
        )

        result = without_center_crop(pre_processor)

        transforms = result.transform.transforms
        assert [type(t) for t in transforms] == [Resize, Normalize]
        assert list(transforms[0].size) == [392, 392]
        assert transforms[0].interpolation == InterpolationMode.BICUBIC
        assert transforms[1] is NORMALIZE

    def test_output_covers_the_whole_image(self) -> None:
        # Mark the image border; it must survive pre-processing instead of being cropped away.
        image = torch.zeros(1, 3, 100, 100)
        image[..., :10, :] = 1.0
        pre_processor = PreProcessor(transform=Compose([Resize((100, 100)), CenterCrop(50)]))

        output = without_center_crop(pre_processor).transform(image)

        assert output.shape[-2:] == (50, 50)
        assert output[..., :5, :].mean() > 0.9

    def test_inserts_resize_when_only_center_cropping(self) -> None:
        pre_processor = PreProcessor(transform=Compose([CenterCrop((64, 96)), NORMALIZE]))

        transforms = without_center_crop(pre_processor).transform.transforms

        assert [type(t) for t in transforms] == [Resize, Normalize]
        assert list(transforms[0].size) == [64, 96]

    def test_dinomaly_default_pre_processor(self) -> None:
        transforms = without_center_crop(Dinomaly.configure_pre_processor()).transform.transforms

        assert not any(isinstance(t, CenterCrop) for t in transforms)
        assert list(transforms[0].size) == [392, 392]

    def test_pre_processor_without_center_crop_is_unchanged(self) -> None:
        pre_processor = PreProcessor(transform=Compose([Resize((256, 256)), NORMALIZE]))

        assert without_center_crop(pre_processor) is pre_processor

    @pytest.mark.parametrize("pre_processor", [None, MagicMock(), PreProcessor(transform=Resize((256, 256)))])
    def test_other_inputs_are_returned_as_is(self, pre_processor) -> None:
        assert without_center_crop(pre_processor) is pre_processor
