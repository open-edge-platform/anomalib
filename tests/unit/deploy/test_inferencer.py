# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for Torch and OpenVINO inferencers."""

from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from anomalib.data import ImageBatch, NumpyImageBatch, PredictDataset
from anomalib.deploy import ExportType, OpenVINOInferencer, TorchInferencer
from anomalib.engine import Engine
from anomalib.models import Padim


class _MockImageLoader:
    """Create mock images for inference on CPU based on the specifics of the original torch test dataset.

    Uses yield so as to avoid storing everything in the memory.

    Args:
        image_size (list[int]): Size of input image
        total_count (int): Total images in the test dataset
    """

    def __init__(self, image_size: list[int], total_count: int, as_numpy: bool = False) -> None:
        self.total_count = total_count
        self.image_size = image_size
        if as_numpy:
            self.image = np.ones((*self.image_size, 3)).astype(np.uint8)
        else:
            self.image = torch.rand((3, *self.image_size))

    def __len__(self) -> int:
        """Get total count of images."""
        return self.total_count

    def __call__(self) -> Iterable[np.ndarray] | Iterable[torch.Tensor]:
        """Yield batch of generated images.

        Args:
            idx (int): Unused
        """
        for _ in range(self.total_count):
            yield self.image


def test_torch_inference(ckpt_path: Callable[[str], Path], monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests Torch inference.

    Model is not trained as this checks that the inferencers are working.

    Args:
        ckpt_path: Callable[[str], Path]: Path to trained PADIM model checkpoint.
        monkeypatch: pytest fixture for patching environment variables.
    """
    # Set TRUST_REMOTE_CODE environment variable for the test
    monkeypatch.setenv("TRUST_REMOTE_CODE", "1")

    model = Padim()
    engine = Engine()
    export_root = ckpt_path("Padim").parent.parent
    engine.export(
        model=model,
        export_type=ExportType.TORCH,
        export_root=export_root,
        ckpt_path=str(ckpt_path("Padim")),
    )
    # Test torch inferencer
    torch_inferencer = TorchInferencer(
        path=export_root / "weights" / "torch" / "model.pt",
        device="cpu",
    )
    torch_dataloader = _MockImageLoader([256, 256], total_count=1)
    with torch.no_grad():
        for image in torch_dataloader():
            prediction = torch_inferencer.predict(image)
            assert 0.0 <= prediction.pred_score <= 1.0  # confirm if predicted scores are normalized


def test_openvino_inference(ckpt_path: Callable[[str], Path]) -> None:
    """Tests OpenVINO inference.

    Model is not trained as this checks that the inferencers are working.

    Args:
        task (TaskType): Task type
        ckpt_path: Callable[[str], Path]: Path to trained PADIM model checkpoint.
        dataset_path (Path): Path to dummy dataset.
    """
    model = Padim()
    engine = Engine()
    export_dir = ckpt_path("Padim").parent.parent
    exported_xml_file_path = engine.export(
        model=model,
        export_type=ExportType.OPENVINO,
        export_root=export_dir,
        ckpt_path=str(ckpt_path("Padim")),
    )

    # Test OpenVINO inferencer
    openvino_inferencer = OpenVINOInferencer(
        exported_xml_file_path,
    )
    openvino_dataloader = _MockImageLoader([256, 256], total_count=1, as_numpy=True)
    for image in openvino_dataloader():
        prediction = openvino_inferencer.predict(image)
        assert 0.0 <= prediction.pred_score <= 1.0  # confirm if predicted scores are normalized


def compare_predictions(
    pred1: ImageBatch | NumpyImageBatch,
    pred2: ImageBatch | NumpyImageBatch,
    tolerance: float = 1e-3,
) -> None:
    """Compare predictions from two different inference methods."""
    score1 = pred1.pred_score if hasattr(pred1, "pred_score") else None
    score2 = pred2.pred_score if hasattr(pred2, "pred_score") else None

    map1 = pred1.anomaly_map if hasattr(pred1, "anomaly_map") else None
    map2 = pred2.anomaly_map if hasattr(pred2, "anomaly_map") else None

    if isinstance(map1, torch.Tensor):
        map1 = map1.cpu().numpy()
    if isinstance(map2, torch.Tensor):
        map2 = map2.cpu().numpy()

    if score1 is None and score2 is None and map1 is None and map2 is None:
        pytest.fail("No predictions found")

    if score1 is not None and score2 is not None:
        if isinstance(score1, torch.Tensor):
            score1 = score1.cpu().item()
        if isinstance(score2, torch.Tensor):
            score2 = score2.cpu().item()

    if score1 is not None and score2 is not None:
        score_diff = abs(score1 - score2)
        if score_diff > tolerance:
            pytest.fail(f"Anomaly score absolute difference: {score_diff:.3f}")

    if map1 is not None and map2 is not None:
        map_diff = np.abs(map1 - map2)
        if np.mean(map_diff) > tolerance:
            pytest.fail(f"Anomaly map mean absolute difference: {np.mean(map_diff):.3f}")


def test_inference_similarity(
    ckpt_path: Callable[[str], Path],
    project_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test inference result."""
    # Set TRUST_REMOTE_CODE environment variable for the test
    monkeypatch.setenv("TRUST_REMOTE_CODE", "1")

    rng = np.random.default_rng()
    image = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
    image = Image.fromarray(image)
    test_image_path = tmp_path / "test_image.png"
    image.save(test_image_path)

    model = Padim()
    engine = Engine(logger=False, default_root_dir=project_path, devices=1)

    predict_dataset = PredictDataset(test_image_path)
    predict_dataloader = DataLoader(
        predict_dataset,
        batch_size=1,
        collate_fn=predict_dataset.collate_fn,
        pin_memory=True,
    )
    engine_pred: list[ImageBatch] = engine.predict(model, dataloaders=predict_dataloader, ckpt_path=ckpt_path("Padim"))
    engine_pred = engine_pred[0]

    torch_path = engine.export(model, export_type=ExportType.TORCH, export_root=project_path)
    torch_inferencer = TorchInferencer(torch_path, device="cpu")
    torch_pred = torch_inferencer.predict(test_image_path)

    openvino_path = engine.export(model, export_type=ExportType.OPENVINO, export_root=project_path)
    openvino_inferencer = OpenVINOInferencer(openvino_path, device="CPU")
    openvino_pred = openvino_inferencer.predict(test_image_path)

    compare_predictions(engine_pred, torch_pred)
    compare_predictions(engine_pred, openvino_pred)
    compare_predictions(torch_pred, openvino_pred)
