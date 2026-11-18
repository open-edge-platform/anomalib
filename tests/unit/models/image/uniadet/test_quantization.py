# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Small offline export checks, distinct from real-data quantization accuracy measurements."""

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from tools.uniadet.data import CalibrationData

from anomalib import TaskType
from anomalib.data import ImageBatch, ImageItem
from anomalib.deploy import CompressionType
from anomalib.metrics import AUROC
from anomalib.models import UniADet

if TYPE_CHECKING:
    import nncf
    import openvino as ov
    from pytest_mock import MockerFixture


@pytest.mark.usefixtures("tiny_backbone")
@pytest.mark.parametrize("compression", [None, *CompressionType])
def test_openvino_compressions(
    tmp_path: Path,
    compression: CompressionType | None,
    mocker: "MockerFixture",
) -> None:
    """Every export path preserves raw scores/maps for singleton and dynamic batches."""
    ov = pytest.importorskip("openvino")
    nncf = pytest.importorskip("nncf")
    quantize = None
    ranking_mode = None
    if compression == CompressionType.INT8_PTQ:
        quantize = mocker.spy(nncf, "quantize")
    elif compression == CompressionType.INT8_ACQ:
        quantize = mocker.spy(nncf, "quantize_with_accuracy_control")
        evaluator = pytest.importorskip("nncf.quantization.algorithms.accuracy_control.evaluator")
        ranking_mode = mocker.spy(evaluator.Evaluator, "determine_mode")
    torch.manual_seed(8)
    model = UniADet(
        pretrained=False,
        image_size=28,
        pre_processor=False,
        evaluator=False,
        visualizer=False,
    ).eval()
    model.model.head_trained.fill_(value=True)
    model.model.set_reference_images(torch.rand(4, 3, 28, 28))
    samples = [
        ImageItem(image=torch.rand(3, 28, 28), gt_label=index % 2, gt_mask=torch.zeros(28, 28)) for index in range(16)
    ]
    data = CalibrationData(samples, subset_size=8)
    path = model.to_openvino(
        tmp_path,
        input_size=(28, 28),
        compression_type=compression,
        datamodule=data,
        metric=AUROC(fields=["pred_score", "gt_label"]),
        task=TaskType.CLASSIFICATION,
        onnx_kwargs={"opset_version": 18},
    )
    compiled = ov.Core().compile_model(path, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    if ranking_mode is not None:
        assert ranking_mode.spy_return is False, "AUROC cannot rank models using single-image AUROCs."
    if quantize is not None:
        assert quantize.call_args.kwargs["model_type"] == nncf.ModelType.TRANSFORMER
        graph = ov.Core().read_model(path)
        for node in graph.get_ordered_ops():
            if node.get_type_name() == "Interpolate":
                assert node.input_value(0).get_node().get_type_name() != "FakeQuantize"
    for batch_size in (1, 3):
        images = torch.stack([item.image for item in samples[:batch_size]])
        with torch.no_grad():
            baseline = model(images)
        actual = compiled(images.numpy())
        tolerance = 1e-4 if compression is None else 0.05
        np.testing.assert_allclose(actual["pred_score"], baseline.pred_score.numpy(), atol=tolerance, rtol=0)
        np.testing.assert_allclose(actual["anomaly_map"], baseline.anomaly_map.numpy(), atol=tolerance, rtol=0)


@pytest.mark.usefixtures("tiny_backbone")
@pytest.mark.parametrize("task", [TaskType.CLASSIFICATION, TaskType.SEGMENTATION])
def test_accuracy_control_resets_metric(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    task: TaskType,
) -> None:
    """Score each candidate independently and return task outputs for non-additive metric ranking."""
    pytest.importorskip("nncf")
    samples = [
        ImageItem(
            image=torch.full((3, 28, 28), float(index % 2)),
            gt_label=index % 2,
            gt_mask=torch.full((28, 28), index % 2),
        )
        for index in range(16)
    ]
    data = CalibrationData(samples, subset_size=8)
    observed = []

    def quantize(
        model: "ov.Model",
        calibration_dataset: "nncf.Dataset",
        validation_dataset: "nncf.Dataset",
        validation_fn: Callable[[Callable, Iterable[ImageBatch]], tuple[float, list[list[NDArray[np.float32]]]]],
        max_drop: float,
        ignored_scope: "nncf.IgnoredScope",
        model_type: "nncf.ModelType",
    ) -> "ov.Model":
        del calibration_dataset, max_drop, ignored_scope, model_type
        inference_items = list(validation_dataset.get_inference_data())
        for batch, images in zip(validation_dataset.get_data(), inference_items, strict=True):
            torch.testing.assert_close(images, batch.image)
        for inverted in (False, True):

            def infer(images: torch.Tensor, inverted: bool = inverted) -> dict[str, np.ndarray]:
                scores = images[:, 0, 0, 0].numpy()
                if inverted:
                    scores = 1 - scores
                return {
                    "pred_score": scores,
                    "pred_label": scores > 0.5,
                    "anomaly_map": np.broadcast_to(scores[:, None, None, None], (len(scores), 1, 28, 28)).copy(),
                }

            metric_value, ranking_outputs = validation_fn(infer, validation_dataset.get_data())
            observed.append(metric_value)
            for output, batch in zip(ranking_outputs, validation_dataset.get_data(), strict=True):
                expected = infer(batch.image)["pred_score" if task == TaskType.CLASSIFICATION else "anomaly_map"]
                np.testing.assert_array_equal(output[0], expected)
        return model

    monkeypatch.setattr("nncf.quantize_with_accuracy_control", quantize)
    model = UniADet(pretrained=False, image_size=28, evaluator=False, visualizer=False).eval()
    model.model.head_trained.fill_(value=True)
    model.to_openvino(
        tmp_path,
        input_size=(28, 28),
        compression_type=CompressionType.INT8_ACQ,
        datamodule=data,
        metric=AUROC(
            fields=["pred_score", "gt_label"] if task == TaskType.CLASSIFICATION else ["anomaly_map", "gt_mask"],
        ),
        task=task,
        onnx_kwargs={"opset_version": 18},
    )
    assert observed == [1.0, 0.0]
