# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Test the PreProcessor class."""

from pathlib import Path

import pytest
import torch
from torchvision.transforms.v2 import Compose, InterpolationMode, Resize, ToDtype, ToImage
from torchvision.tv_tensors import Image, Mask

from anomalib.data import ImageBatch
from anomalib.pre_processing import PreProcessor


class TestPreProcessor:
    """Test the PreProcessor class."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Set up test fixtures for each test method."""
        image = Image(torch.rand(3, 256, 256))
        gt_mask = Mask(torch.zeros(256, 256))
        self.dummy_batch = ImageBatch(image=image, gt_mask=gt_mask)
        self.common_transform = Compose([Resize((224, 224)), ToImage(), ToDtype(torch.float32, scale=True)])

    def test_forward(self) -> None:
        """Test the forward method of the PreProcessor class."""
        pre_processor = PreProcessor(transform=self.common_transform)
        processed_batch = pre_processor(self.dummy_batch.image)
        assert isinstance(processed_batch, torch.Tensor)
        assert processed_batch.shape == (1, 3, 224, 224)

    def test_no_transform(self) -> None:
        """Test no transform."""
        pre_processor = PreProcessor()
        processed_batch = pre_processor(self.dummy_batch.image)
        assert isinstance(processed_batch, torch.Tensor)
        assert processed_batch.shape == (1, 3, 256, 256)

    @staticmethod
    @pytest.mark.parametrize("interpolation", [InterpolationMode.BILINEAR, InterpolationMode.BICUBIC])
    @pytest.mark.parametrize("input_size", [(160, 200), (40, 48)])
    def test_onnx_export_matches_predict_transform(
        tmp_path: Path,
        interpolation: InterpolationMode,
        input_size: tuple[int, int],
    ) -> None:
        """The exported pre-processing resizes like the transform used by ``Engine.predict``."""
        onnx = pytest.importorskip("onnx")
        pytest.importorskip("onnxscript")
        from onnx.reference import ReferenceEvaluator

        pre_processor = PreProcessor(transform=Resize((64, 64), interpolation=interpolation, antialias=True))
        image = torch.rand(1, 3, *input_size)
        onnx_path = tmp_path / "pre_processor.onnx"
        torch.onnx.export(pre_processor, (image,), onnx_path, dynamo=True)

        model = onnx.load(onnx_path)
        exported = ReferenceEvaluator(model).run(None, {model.graph.input[0].name: image.numpy()})[0]
        expected = pre_processor.transform(image)
        torch.testing.assert_close(torch.from_numpy(exported), expected, atol=1e-5, rtol=1e-5)
