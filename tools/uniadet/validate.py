# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Numerical deployment validation for a trained UniADet head and fixed target category.

Runs TorchInferencer, Engine, ONNX Runtime, and OpenVINO independently. Quantized
models are compared against raw FP32 scores/maps, not merely checked for valid shapes.
"""

import argparse
import gc
import json
import logging
import os
import platform
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import onnxruntime as ort
import openvino as ov
import torch
from lightning.pytorch import Callback
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Resize

from anomalib import TaskType
from anomalib.callbacks import ModelCheckpoint
from anomalib.data import ImageBatch, ImageItem
from anomalib.data.datasets.image.mvtecad import CATEGORIES as MVTEC_CATEGORIES
from anomalib.data.datasets.image.mvtecad import MVTecADDataset
from anomalib.data.datasets.image.visa import CATEGORIES as VISA_CATEGORIES
from anomalib.data.datasets.image.visa import VisaDataset
from anomalib.deploy import CompressionType, TorchInferencer
from anomalib.engine import Engine
from anomalib.metrics import AUROC
from anomalib.models import UniADet

from .data import AuxiliaryDataset, CalibrationData

if TYPE_CHECKING:
    from lightning.pytorch import LightningModule, Trainer

logger = logging.getLogger(__name__)

TOLERANCES = {
    "torch_inferencer": 1e-5,
    "engine": 1e-5,
    "onnx": 1e-4,
    "fp32": 1e-4,
    "fp16": 0.005,
    "int8": 0.05,
    "int8_ptq": 0.05,
    "int8_acq": 0.05,
}


def compare_outputs(
    baseline: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    tolerance: float,
) -> dict[str, dict[str, float | bool]]:
    """Compare finite raw scores and maps, preserving batch and spatial dimensions.

    Args:
        baseline (dict[str, np.ndarray]): PyTorch FP32 score/map arrays.
        candidate (dict[str, np.ndarray]): Deployment output arrays.
        tolerance (float): Predeclared absolute maximum error tolerance.

    Returns:
        dict[str, dict[str, float | bool]]: Per-output error measurements and pass flags.

    Raises:
        ValueError: If output shapes do not match.
    """
    results = {}
    for name in ("pred_score", "anomaly_map"):
        expected, actual = baseline[name], candidate[name]
        if expected.shape != actual.shape:
            msg = f"{name} shape differs: {expected.shape} versus {actual.shape}."
            raise ValueError(msg)
        error = np.abs(actual - expected)
        finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
        results[name] = {
            "max_absolute_error": float(error.max()),
            "mean_absolute_error": float(error.mean()),
            "rmse": float(np.sqrt(np.mean(error**2))),
            "tolerance": tolerance,
            "finite": finite,
            "passed": finite and bool(error.max() <= tolerance),
        }
    return results


def _reference_images(
    model: UniADet,
    dataset: MVTecADDataset | VisaDataset,
    shots: int,
) -> None:
    normal = dataset.samples.index[dataset.samples.label_index == 0].tolist()
    if len(normal) < shots:
        msg = f"Requested {shots} references but only {len(normal)} normal images are available."
        raise ValueError(msg)
    order = torch.randperm(len(normal), generator=torch.Generator().manual_seed(10))[:shots]
    references = torch.stack([dataset[normal[int(index)]].image for index in order])
    if model.pre_processor is None:
        msg = "Reference initialization requires backbone preprocessing."
        raise ValueError(msg)
    model.model.set_reference_images(model.pre_processor(references))


def _prediction_arrays(prediction: ImageBatch) -> dict[str, np.ndarray]:
    if prediction.pred_score is None or prediction.anomaly_map is None:
        msg = "Inference returned no scores or maps."
        raise RuntimeError(msg)
    return {
        "pred_score": prediction.pred_score.detach().cpu().numpy(),
        "anomaly_map": prediction.anomaly_map.detach().cpu().unsqueeze(1).numpy(),
    }


class _PredictionCapture(Callback):
    """Capture raw Engine test outputs without replacing its inference lifecycle."""

    def __init__(self) -> None:
        self.predictions: list[dict[str, np.ndarray]] = []

    def on_test_batch_end(
        self,
        trainer: "Trainer",
        pl_module: "LightningModule",
        outputs: ImageBatch,
        batch: ImageBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Retain scores and maps for independent numerical comparison."""
        del trainer, pl_module, batch, batch_idx, dataloader_idx
        self.predictions.append(_prediction_arrays(outputs))


def _engine_predictions(
    model: UniADet,
    batches: dict[int, torch.Tensor],
    output: Path,
) -> tuple[dict[int, dict[str, np.ndarray]], dict[int, dict[str, np.ndarray]]]:
    """Collect independent Engine predict/test results after saving and restoring a local checkpoint."""
    capture = _PredictionCapture()
    engine = Engine(
        accelerator="cpu",
        devices=1,
        logger=False,
        callbacks=[ModelCheckpoint(save_top_k=0), capture],
        enable_progress_bar=False,
        default_root_dir=output,
    )
    images = batches[max(batches)]
    samples = [ImageItem(image=image, gt_label=index > 0) for index, image in enumerate(images)]
    loader = DataLoader(samples, batch_size=1, collate_fn=ImageBatch.collate)
    engine.fit(model, val_dataloaders=loader)
    checkpoint = output / "engine.ckpt"
    engine.trainer.save_checkpoint(checkpoint)
    predictions, test_predictions = {}, {}
    for size in batches:
        loader = DataLoader(samples[:size], batch_size=size, collate_fn=ImageBatch.collate)
        result = engine.predict(model, dataloaders=loader, ckpt_path=checkpoint, return_predictions=True)
        if not result:
            msg = "Engine returned no predictions."
            raise RuntimeError(msg)
        predictions[size] = _prediction_arrays(result[0])
        capture.predictions.clear()
        engine.test(model, dataloaders=loader, ckpt_path=checkpoint)
        if len(capture.predictions) != 1:
            msg = "Engine.test did not return exactly one batch for numerical comparison."
            raise RuntimeError(msg)
        test_predictions[size] = capture.predictions[0]
    return predictions, test_predictions


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--target", choices=("mvtec", "visa"), required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--auxiliary-root", type=Path, required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--shots", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-size", type=int, default=300)
    parser.add_argument("--formats", nargs="+", choices=list(TOLERANCES), default=list(TOLERANCES))
    args = parser.parse_args()
    allowed_categories = MVTEC_CATEGORIES if args.target == "mvtec" else VISA_CATEGORIES
    if args.category not in allowed_categories:
        msg = f"The category must belong to {tuple(allowed_categories)}."
        raise ValueError(msg)
    if args.shots < 0:
        msg = "The reference count must be nonnegative."
        raise ValueError(msg)
    return args


@torch.inference_mode()
def _measure_latency(
    predict: Callable[[torch.Tensor], object],
    batches: dict[int, torch.Tensor],
) -> dict[int, dict[str, float]]:
    measurements = {}
    for size, images in batches.items():
        predict(images)
        durations = []
        for _ in range(3):
            start = time.perf_counter()
            predict(images)
            durations.append(time.perf_counter() - start)
        median = statistics.median(durations)
        measurements[size] = {
            "median_batch_seconds": median,
            "images_per_second": size / median,
        }
    return measurements


def main() -> None:
    """Export a category-specific detector and write measured parity results for batches 1 and 3."""
    args = _parse_args()
    logging.basicConfig(level=logging.INFO)
    torch.set_num_threads(4)
    manifest = json.loads((args.weights.parent / "training.json").read_text())
    if manifest.get("recipe_version") != 2:
        msg = "Retrain with corrected recipe version 2 before collecting final deployment measurements."
        raise ValueError(msg)
    if manifest["source_dataset"] == args.target:
        msg = "Calibration source and target dataset must differ."
        raise ValueError(msg)
    model = UniADet(
        backbone=manifest["backbone"],
        weights=args.weights,
        k_shot=args.shots,
        evaluator=False,
        visualizer=False,
    ).eval()
    dataset_class = MVTecADDataset if args.target == "mvtec" else VisaDataset
    resize = Resize((model.image_size, model.image_size), antialias=False)
    target_data = dataset_class(root=args.target_root, category=args.category, split="test", augmentations=resize)
    if args.shots:
        reference_data = dataset_class(
            root=args.target_root,
            category=args.category,
            split="train",
            augmentations=resize,
        )
        _reference_images(model, reference_data, args.shots)
    normal = target_data.samples.index[target_data.samples.label_index == 0].tolist()
    anomalous = target_data.samples.index[target_data.samples.label_index == 1].tolist()
    selected = (normal[:1] + anomalous[:2]) if normal and len(anomalous) >= 2 else list(range(3))
    if len(target_data) < 3:
        msg = "Numerical batch validation requires at least three target images."
        raise ValueError(msg)
    images = torch.stack([target_data[index].image for index in selected])
    batches = {size: images[:size] for size in (1, 3)}
    with torch.inference_mode():
        baselines = {
            size: {
                name: value.numpy()
                for name, value in model(batch)._asdict().items()
                if name in {"pred_score", "anomaly_map"}
            }
            for size, batch in batches.items()
        }
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "backbone": manifest["backbone"],
        "head_sha256": manifest["head_sha256"],
        "training_recipe_version": manifest["recipe_version"],
        "complete_training_recipe": manifest["complete_recipe"],
        "target": args.target,
        "category": args.category,
        "shots": args.shots,
        "images": [str(target_data.samples.iloc[index].image_path) for index in selected],
        "calibration_source": manifest["source_dataset"],
        "calibration_size": args.calibration_size,
        "quantized_max_drop": 0.01,
        "activation_quantization": {
            "model_type": "TRANSFORMER",
            "floating_point_postprocessing": ["Interpolate", "Gaussian smoothing"],
            "accuracy_control_ranking": "continuous_task_outputs",
        },
        "timing_protocol": {
            "device": "CPU",
            "threads": 4,
            "warmup": 1,
            "repeats": 3,
            "includes": "In-memory model preprocessing, forward and postprocessing; excludes decoding and compilation.",
            "isolation": "Shared development host; other training/compilation jobs may be running.",
            "host": platform.platform(),
            "torch": torch.__version__,
            "openvino": ov.__version__,
            "onnxruntime": ort.__version__,
        },
        "pytorch_cpu_timing": _measure_latency(model, batches),
        "results": {},
    }
    calibration = CalibrationData(
        AuxiliaryDataset(args.auxiliary_root, manifest["source_dataset"], model.image_size, augmentation=False),
        args.calibration_size,
    )
    for format_name in args.formats:
        start = time.perf_counter()
        outputs = {}
        timing = {}
        if format_name == "torch_inferencer":
            path = model.to_torch(args.output / format_name)
            # Trust only the file generated by this process; never enable this for downloaded files.
            previous = os.environ.get("TRUST_REMOTE_CODE")
            os.environ["TRUST_REMOTE_CODE"] = "1"
            try:
                inferencer = TorchInferencer(path, device="cpu")
            finally:
                if previous is None:
                    del os.environ["TRUST_REMOTE_CODE"]
                else:
                    os.environ["TRUST_REMOTE_CODE"] = previous
            for size, batch in batches.items():
                prediction = inferencer.predict(batch[0] if size == 1 else batch)
                outputs[size] = _prediction_arrays(prediction)
            file_start = time.perf_counter()
            file_prediction = inferencer.predict(Path(target_data.samples.iloc[selected[0]].image_path))
            report["results"]["torch_file"] = {
                "duration_seconds": time.perf_counter() - file_start,
                "batches": {
                    1: compare_outputs(
                        baselines[1],
                        _prediction_arrays(file_prediction),
                        TOLERANCES["torch_inferencer"],
                    ),
                },
            }
            timing = _measure_latency(inferencer.predict, batches)
            del inferencer
        elif format_name == "engine":
            outputs, test_predictions = _engine_predictions(model, batches, args.output)
            report["results"]["engine_test"] = {
                "batches": {
                    size: compare_outputs(baselines[size], test_predictions[size], TOLERANCES["engine"])
                    for size in batches
                },
            }
        elif format_name == "onnx":
            path = model.to_onnx(args.output / format_name, input_size=(model.image_size,) * 2, opset_version=18)
            options = ort.SessionOptions()
            options.intra_op_num_threads = 4
            session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
            names = [output.name for output in session.get_outputs()]
            outputs = {
                size: dict(zip(names, session.run(None, {"input": batch.numpy()}), strict=True))
                for size, batch in batches.items()
            }
            timing = _measure_latency(
                lambda images, session=session: session.run(None, {"input": images.numpy()}),
                batches,
            )
            del session
        else:
            compression = None if format_name == "fp32" else CompressionType(format_name)
            path = model.to_openvino(
                args.output / format_name,
                input_size=(model.image_size,) * 2,
                compression_type=compression,
                datamodule=calibration,
                metric=AUROC(fields=["pred_score", "gt_label"]),
                task=TaskType.CLASSIFICATION,
                max_drop=0.01,
                onnx_kwargs={"opset_version": 18},
            )
            compiled = ov.Core().compile_model(
                path,
                "CPU",
                {"INFERENCE_PRECISION_HINT": "f32", "INFERENCE_NUM_THREADS": 4},
            )
            for size, batch in batches.items():
                outputs[size] = {port.any_name: value for port, value in compiled(batch.numpy()).items()}
            timing = _measure_latency(lambda images, compiled=compiled: compiled(images.numpy()), batches)
            del compiled
        report["results"][format_name] = {
            "duration_seconds": time.perf_counter() - start,
            "inference_timing": timing,
            "batches": {
                size: compare_outputs(baselines[size], outputs[size], TOLERANCES[format_name]) for size in batches
            },
        }
        (args.output / "parity.json").write_text(json.dumps(report, indent=2) + "\n")
        logger.info("%s: %s", format_name, report["results"][format_name])
        gc.collect()
    passed = all(
        output["passed"]
        for result in report["results"].values()
        for batch in result["batches"].values()
        for output in batch.values()
    )
    if not passed:
        msg = f"One or more numerical parity checks failed; see {args.output / 'parity.json'}."
        raise RuntimeError(msg)


if __name__ == "__main__":
    main()
