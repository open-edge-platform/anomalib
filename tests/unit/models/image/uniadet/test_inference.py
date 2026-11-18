# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Inference lifecycle and deployment tests with a deterministic offline encoder."""

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from anomalib.data import ImageBatch, ImageItem
from anomalib.deploy import TorchInferencer
from anomalib.engine import Engine
from anomalib.models import UniADet


@pytest.mark.usefixtures("tiny_backbone")
@pytest.mark.parametrize("shots", [0, 4])
def test_engine_torch_and_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shots: int) -> None:
    """Direct, exported TorchInferencer, Engine, and checkpoint predictions must agree."""
    torch.manual_seed(7)
    model = UniADet(pretrained=False, image_size=28, k_shot=shots, evaluator=False, visualizer=False)
    model.model.head_trained.fill_(value=True)
    if shots:
        model.model.set_reference_images(model.pre_processor(torch.rand(shots, 3, 28, 28)))
    images = torch.rand(3, 3, 28, 28)
    model.eval()
    with torch.no_grad():
        expected = model(images)
    engine = Engine(
        accelerator="cpu",
        default_root_dir=tmp_path,
        logger=False,
        enable_progress_bar=False,
    )
    loader = DataLoader(
        [ImageItem(image=image, gt_label=index % 2) for index, image in enumerate(images)],
        batch_size=2,
        collate_fn=ImageBatch.collate,
    )
    engine.fit(model, val_dataloaders=loader)
    checkpoint = tmp_path / "model.ckpt"
    engine.trainer.save_checkpoint(checkpoint)
    predictions = engine.predict(model, dataloaders=loader, return_predictions=True, ckpt_path=checkpoint)
    torch.testing.assert_close(torch.cat([batch.pred_score for batch in predictions]), expected.pred_score)
    torch.testing.assert_close(torch.cat([batch.anomaly_map for batch in predictions]), expected.anomaly_map.squeeze(1))

    restored = UniADet.load_from_checkpoint(checkpoint)
    with torch.no_grad():
        torch.testing.assert_close(restored(images).pred_score, expected.pred_score)

    path = model.to_torch(tmp_path)
    # Only trust the export produced locally by this test, never a downloaded pickle.
    monkeypatch.setenv("TRUST_REMOTE_CODE", "1")
    inferencer = TorchInferencer(path, device="cpu")
    batch = inferencer.predict(images)
    torch.testing.assert_close(batch.pred_score, expected.pred_score)
    for index, image in enumerate(images):
        prediction = inferencer.predict(image)
        torch.testing.assert_close(prediction.pred_score.flatten(), expected.pred_score[index : index + 1])


@pytest.mark.usefixtures("tiny_backbone")
@pytest.mark.parametrize("shots", [0, 4])
def test_onnx_dynamic_batch(tmp_path: Path, shots: int) -> None:
    """ONNX graph executes independently for both singleton and larger dynamic batches."""
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")
    model = UniADet(pretrained=False, image_size=28, k_shot=shots, evaluator=False, visualizer=False).eval()
    model.model.head_trained.fill_(value=True)
    if shots:
        model.model.set_reference_images(model.pre_processor(torch.rand(shots, 3, 28, 28)))
    path = model.to_onnx(tmp_path, input_size=(28, 28), opset_version=18)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    names = [output.name for output in session.get_outputs()]
    for batch_size in (1, 3):
        images = torch.rand(batch_size, 3, 28, 28)
        with torch.no_grad():
            baseline = model(images)
        outputs = dict(zip(names, session.run(None, {"input": images.numpy()}), strict=True))
        np.testing.assert_allclose(outputs["pred_score"], baseline.pred_score.numpy(), rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(outputs["anomaly_map"], baseline.anomaly_map.numpy(), rtol=1e-4, atol=1e-5)
