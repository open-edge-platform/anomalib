# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Inference lifecycle and deployment tests with a deterministic offline encoder."""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from tools.uniadet import validate
from tools.uniadet.validate import _engine_predictions
from torch.utils.data import DataLoader

from anomalib.callbacks import ModelCheckpoint
from anomalib.data import ImageBatch, ImageItem
from anomalib.data.datasets.image.mvtecad import MVTecADDataset
from anomalib.data.utils import read_image
from anomalib.deploy import TorchInferencer
from anomalib.engine import Engine
from anomalib.models import UniADet


@pytest.mark.parametrize("backend", ["onnxruntime", "openvino"])
def test_torch_validation_without_export_backend(monkeypatch: pytest.MonkeyPatch, backend: str) -> None:
    """Torch-only validation must remain importable without optional export backends."""
    monkeypatch.setitem(sys.modules, backend, None)
    spec = importlib.util.spec_from_file_location("tools.uniadet.validate", validate.__file__)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


@pytest.mark.usefixtures("tiny_backbone")
@pytest.mark.parametrize("protocol", ["legacy", "author"])
def test_validation_reference_sampling(protocol: str) -> None:
    """Author sampling permits repeated normal references; legacy sampling does not."""
    model = UniADet(pretrained=False, image_size=28, k_shot=4, protocol=protocol)
    dataset = MagicMock(spec=MVTecADDataset)
    dataset.samples = pd.DataFrame({"label_index": [0]})
    dataset.__getitem__.return_value = ImageItem(image=torch.rand(3, 28, 28))
    if protocol == "legacy":
        with pytest.raises(ValueError, match="only 1 normal"):
            validate._reference_images(model, dataset, 4)  # noqa: SLF001
    else:
        validate._reference_images(model, dataset, 4)  # noqa: SLF001
        assert dataset.__getitem__.call_count == 4
        references = model.model.memory_bank.chunk(4, dim=1)
        for reference in references[1:]:
            torch.testing.assert_close(reference, references[0])

    dataset.samples["label_index"] = 1
    with pytest.raises(ValueError, match="only 0 normal"):
        validate._reference_images(model, dataset, 4)  # noqa: SLF001


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
        callbacks=[ModelCheckpoint(save_top_k=0)],
        enable_progress_bar=False,
    )
    loader = DataLoader(
        [ImageItem(image=image, gt_label=index % 2) for index, image in enumerate(images)],
        batch_size=2,
        collate_fn=ImageBatch.collate,
    )
    engine.fit(model, val_dataloaders=loader)
    assert len(engine.trainer.checkpoint_callbacks) == 1
    assert not list(tmp_path.rglob("*.ckpt"))
    checkpoint = tmp_path / "model.ckpt"
    engine.trainer.save_checkpoint(checkpoint)
    with torch.no_grad():
        calibrated = model(images)
    predictions = engine.predict(model, dataloaders=loader, return_predictions=True, ckpt_path=checkpoint)
    torch.testing.assert_close(torch.cat([batch.pred_score for batch in predictions]), expected.pred_score)
    torch.testing.assert_close(torch.cat([batch.anomaly_map for batch in predictions]), expected.anomaly_map.squeeze(1))
    torch.testing.assert_close(torch.cat([batch.pred_label for batch in predictions]), calibrated.pred_label)
    torch.testing.assert_close(torch.cat([batch.pred_mask for batch in predictions]), calibrated.pred_mask.squeeze(1))
    restored = UniADet.load_from_checkpoint(checkpoint)
    with torch.no_grad():
        torch.testing.assert_close(restored(images).pred_score, expected.pred_score)
    validation_root = tmp_path / "engine_validation"
    validation_root.mkdir()
    predicted, tested = _engine_predictions(restored, {1: images[:1], 3: images}, validation_root)
    for outputs in (predicted, tested):
        for size in (1, 3):
            np.testing.assert_allclose(outputs[size]["pred_score"], expected.pred_score[:size].numpy(), atol=1e-5)
            np.testing.assert_allclose(outputs[size]["anomaly_map"], expected.anomaly_map[:size].numpy(), atol=1e-5)

    path = model.to_torch(tmp_path)
    # Only trust the export produced locally by this test, never a downloaded pickle.
    monkeypatch.setenv("TRUST_REMOTE_CODE", "1")
    inferencer = TorchInferencer(path, device="cpu")
    batch = inferencer.predict(images)
    torch.testing.assert_close(batch.pred_score, expected.pred_score)
    torch.testing.assert_close(batch.pred_label, calibrated.pred_label)
    torch.testing.assert_close(batch.pred_mask, calibrated.pred_mask.squeeze(1))
    for index, image in enumerate(images):
        prediction = inferencer.predict(image)
        torch.testing.assert_close(prediction.pred_score.flatten(), expected.pred_score[index : index + 1])
    image_path = tmp_path / "query.png"
    Image.fromarray((images[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)).resize((41, 37)).save(image_path)
    with torch.no_grad():
        file_expected = model(read_image(image_path, as_tensor=True).unsqueeze(0))
    file_prediction = inferencer.predict(image_path)
    torch.testing.assert_close(file_prediction.pred_score, file_expected.pred_score)
    torch.testing.assert_close(file_prediction.anomaly_map, file_expected.anomaly_map.squeeze(1))


@pytest.mark.usefixtures("tiny_backbone")
@pytest.mark.parametrize("overrides", [{"k_shot": 0}, {"k_shot": 1}, {"seed": 20}, {"few_shot_source": "other"}])
def test_checkpoint_rejects_changed_references(tmp_path: Path, overrides: dict[str, int | str]) -> None:
    """A full checkpoint must not silently reuse a bank with different reference settings."""
    model = UniADet(pretrained=False, image_size=28, k_shot=4, evaluator=False, visualizer=False)
    model.model.head_trained.fill_(value=True)
    model.model.set_reference_images(model.pre_processor(torch.rand(4, 3, 28, 28)))
    _engine_predictions(model, {1: torch.rand(1, 3, 28, 28)}, tmp_path)
    with pytest.raises(ValueError, match="reference configuration"):
        UniADet.load_from_checkpoint(tmp_path / "engine.ckpt", **overrides)


@pytest.mark.usefixtures("tiny_backbone")
@pytest.mark.parametrize("shots", [0, 4])
def test_onnx_dynamic_batch(tmp_path: Path, shots: int) -> None:
    """ONNX graph executes independently for both singleton and larger dynamic batches."""
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")
    model = UniADet(pretrained=False, image_size=28, k_shot=shots, evaluator=False, visualizer=False).eval()
    model.model.head_trained.fill_(value=True)
    model.post_processor._image_threshold.fill_(0.8)  # noqa: SLF001
    model.post_processor._pixel_threshold.fill_(0.4)  # noqa: SLF001
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
        np.testing.assert_array_equal(outputs["pred_label"], baseline.pred_label.numpy())
        np.testing.assert_array_equal(outputs["pred_mask"], baseline.pred_mask.numpy())
