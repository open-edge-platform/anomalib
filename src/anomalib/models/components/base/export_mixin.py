# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Mixin for exporting anomaly detection models to disk.

This mixin provides functionality to export models to various formats:
- PyTorch (.pt)
- ONNX (.onnx)
- OpenVINO IR (.xml/.bin)

The mixin supports different compression types for OpenVINO exports:
- FP16 compression
- INT8 quantization
- Post-training quantization (PTQ)
- Accuracy-aware quantization (ACQ)

Example:
    Export a trained model to different formats:

    >>> from anomalib.models import Patchcore
    >>> from anomalib.data import Visa
    >>> from anomalib.deploy.export import CompressionType
    ...
    >>> # Initialize and train model
    >>> model = Patchcore()
    >>> datamodule = Visa()
    >>> # Export to PyTorch format
    >>> model.to_torch("./exports")
    >>> # Export to ONNX
    >>> model.to_onnx("./exports", input_size=(224, 224))
    >>> # Export to OpenVINO with INT8 quantization
    >>> model.to_openvino(
    ...     "./exports",
    ...     input_size=(224, 224),
    ...     compression_type=CompressionType.INT8_PTQ,
    ...     datamodule=datamodule
    ... )
"""

import logging
from collections.abc import Iterable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

import torch
from lightning.pytorch import LightningModule
from lightning_utilities.core.imports import module_available
from torch import nn
from torchmetrics import Metric

from anomalib import TaskType
from anomalib.data import AnomalibDataModule
from anomalib.deploy.export import CompressionType, ExportType

if TYPE_CHECKING:
    if module_available("openvino"):
        from openvino import CompiledModel

logger = logging.getLogger(__name__)


class ExportMixin:
    """Mixin class that enables exporting models to various formats.

    This mixin provides methods to export models to PyTorch (.pt), ONNX (.onnx),
    and OpenVINO IR (.xml/.bin) formats. For OpenVINO exports, it supports
    different compression types including FP16, INT8, PTQ and ACQ.

    The mixin requires the host class to have:
        - A ``model`` attribute of type ``nn.Module``
        - A ``device`` attribute of type ``torch.device``
    """

    model: nn.Module
    device: torch.device

    def to_torch(
        self,
        export_root: Path | str,
        model_file_name: str = "model",
    ) -> Path:
        """Export model to PyTorch format.

        Args:
            export_root (Path | str): Path to the output folder
            model_file_name (str): Name of the exported model

        Returns:
            Path: Path to the exported PyTorch model (.pt file)

        Examples:
            Export a trained model to PyTorch format:

            >>> from anomalib.models import Patchcore
            >>> model = Patchcore()
            >>> # Train model...
            >>> model.to_torch("./exports")
            PosixPath('./exports/weights/torch/model.pt')
        """
        export_root = _create_export_root(export_root, ExportType.TORCH)
        pt_model_path = export_root / (model_file_name + ".pt")
        # See mitigation details in https://github.com/open-edge-platform/anomalib/pull/2729
        # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
        torch.save(
            obj={"model": self},
            f=pt_model_path,
        )
        return pt_model_path

    def to_onnx(
        self,
        export_root: Path | str,
        model_file_name: str = "model",
        input_size: tuple[int, int] | None = None,
        **kwargs,
    ) -> Path:
        """Export model to ONNX format.

        Args:
            export_root (Path | str): Path to the output folder
            model_file_name (str): Name of the exported model.
            input_size (tuple[int, int] | None): Input image dimensions (height, width).
                If ``None``, uses dynamic input shape. Defaults to ``None``
            **kwargs: Additional arguments to pass to torch.onnx.export.
                See https://pytorch.org/docs/stable/onnx.html#torch.onnx.export for details.
                Common options include:
                - opset_version (int): ONNX opset version to use
                - do_constant_folding (bool): Whether to optimize constant folding
                - input_names (list[str]): Names of input tensors
                - output_names (list[str]): Names of output tensors
                - dynamic_axes (dict): Dynamic axes configuration
                - custom_opsets (dict): Custom opset versions
                - export_modules_as_functions (bool): Export modules as functions
                - verify (bool): Verify the exported model
                - optimize (bool): Optimize the exported model

        Returns:
            Path: Path to the exported ONNX model (.onnx file)

        Examples:
            Export model with fixed input size:

            >>> from anomalib.models import Patchcore
            >>> model = Patchcore()
            >>> # Train model...
            >>> model.to_onnx("./exports", input_size=(224, 224))
            PosixPath('./exports/weights/onnx/model.onnx')

            Export model with custom options:

            >>> model.to_onnx(
            ...     "./exports",
            ...     opset_version=12,
            ...     do_constant_folding=True,
            ...     verify=True,
            ...     optimize=True
            ... )
            PosixPath('./exports/weights/onnx/model.onnx')
        """
        export_root = _create_export_root(export_root, ExportType.ONNX)
        input_shape = torch.zeros((1, 3, *input_size)) if input_size else torch.zeros((1, 3, 1, 1))
        input_shape = input_shape.to(self.device)
        dynamic_axes = (
            {"input": {0: "batch_size"}, "output": {0: "batch_size"}}
            if input_size
            else {"input": {0: "batch_size", 2: "height", 3: "width"}, "output": {0: "batch_size"}}
        )
        onnx_path = export_root / (model_file_name + ".onnx")
        # apply pass through the model to get the output names
        assert isinstance(self, LightningModule)  # mypy
        output_names = [name for name, value in self.eval()(input_shape)._asdict().items() if value is not None]

        torch.onnx.export(
            model=self,
            args=(input_shape.to(self.device),),
            f=str(onnx_path),
            opset_version=kwargs.pop("opset_version", 14),
            dynamic_axes=kwargs.pop("dynamic_axes", dynamic_axes),
            input_names=kwargs.pop("input_names", ["input"]),
            output_names=kwargs.pop("output_names", output_names),
            **kwargs,
        )

        return onnx_path

    def to_openvino(
        self,
        export_root: Path | str,
        model_file_name: str = "model",
        input_size: tuple[int, int] | None = None,
        compression_type: CompressionType | None = None,
        datamodule: AnomalibDataModule | None = None,
        metric: Metric | None = None,
        task: TaskType | None = None,
        ov_kwargs: dict[str, Any] | None = None,
        onnx_kwargs: dict[str, Any] | None = None,
    ) -> Path:
        """Export model to OpenVINO IR format.

        Args:
            export_root (Path | str): Path to the output folder
            model_file_name (str): Name of the exported model
            input_size (tuple[int, int] | None): Input image dimensions (height, width).
                If ``None``, uses dynamic input shape. Defaults to ``None``
            compression_type (CompressionType | None): Type of compression to apply.
                Options: ``FP16``, ``INT8``, ``INT8_PTQ``, ``INT8_ACQ``.
                Defaults to ``None``
            datamodule (AnomalibDataModule | None): DataModule for quantization.
                Required for ``INT8_PTQ`` and ``INT8_ACQ``. Defaults to ``None``
            metric (Metric | None): Metric for accuracy-aware quantization.
                Required for ``INT8_ACQ``. Defaults to ``None``
            task (TaskType | None): Task type (classification/segmentation).
                Defaults to ``None``
            ov_kwargs (dict[str, Any] | None): OpenVINO model optimizer arguments.
                Defaults to ``None``
            onnx_kwargs (dict[str, Any] | None): Additional arguments to pass to torch.onnx.export
                during the initial ONNX conversion. See https://pytorch.org/docs/stable/onnx.html#torch.onnx.export
                for details. Defaults to ``None``

        Returns:
            Path: Path to the exported OpenVINO model (.xml file)

        Raises:
            ModuleNotFoundError: If OpenVINO is not installed
            ValueError: If required arguments for quantization are missing

        Examples:
            Export model with FP16 compression:

            >>> model.to_openvino(
            ...     "./exports",
            ...     input_size=(224, 224),
            ...     compression_type=CompressionType.FP16
            ... )

            Export with INT8 post-training quantization and custom options:

            >>> model.to_openvino(
            ...     "./exports",
            ...     compression_type=CompressionType.INT8_PTQ,
            ...     datamodule=datamodule,
            ...     ov_kwargs={"input_shape": [1, 3, 224, 224]},
            ...     onnx_kwargs={"opset_version": 12, "do_constant_folding": True}
            ... )
        """
        if not module_available("openvino"):
            logger.exception("Could not find OpenVINO. Please check OpenVINO installation.")
            raise ModuleNotFoundError

        import openvino as ov

        with TemporaryDirectory() as onnx_directory:
            model_path = self.to_onnx(onnx_directory, model_file_name, input_size, **(onnx_kwargs or {}))
            export_root = _create_export_root(export_root, ExportType.OPENVINO)
            ov_model_path = export_root / (model_file_name + ".xml")

            model = ov.convert_model(model_path, **(ov_kwargs or {}))
            if compression_type and compression_type != CompressionType.FP16:
                model = self._compress_ov_model(model, compression_type, datamodule, metric, task)

            # fp16 compression is enabled by default
            compress_to_fp16 = compression_type == CompressionType.FP16
            ov.save_model(model, ov_model_path, compress_to_fp16=compress_to_fp16)

        return ov_model_path

    def _compress_ov_model(
        self,
        model: "CompiledModel",
        compression_type: CompressionType | None = None,
        datamodule: AnomalibDataModule | None = None,
        metric: Metric | None = None,
        task: TaskType | None = None,
    ) -> "CompiledModel":
        """Compress OpenVINO model using NNCF.

        Args:
            model (CompiledModel): OpenVINO model to compress
            compression_type (CompressionType | None): Type of compression to apply.
                Defaults to ``None``
            datamodule (AnomalibDataModule | None): DataModule for quantization.
                Required for ``INT8_PTQ`` and ``INT8_ACQ``. Defaults to ``None``
            metric (Metric | None): Metric for accuracy-aware quantization.
                Required for ``INT8_ACQ``. Defaults to ``None``
            task (TaskType | None): Task type (classification/segmentation).
                Defaults to ``None``

        Returns:
            CompiledModel: Compressed OpenVINO model

        Raises:
            ModuleNotFoundError: If NNCF is not installed
            ValueError: If compression type is not recognized
        """
        if not module_available("nncf"):
            logger.exception("Could not find NCCF. Please check NNCF installation.")
            raise ModuleNotFoundError

        import nncf

        if compression_type == CompressionType.INT8:
            model = nncf.compress_weights(model)
        elif compression_type == CompressionType.INT8_PTQ:
            model = self._post_training_quantization_ov(model, datamodule)
        elif compression_type == CompressionType.INT8_ACQ:
            model = self._accuracy_control_quantization_ov(model, datamodule, metric, task)
        else:
            msg = f"Unrecognized compression type: {compression_type}"
            raise ValueError(msg)

        return model

    @staticmethod
    def _post_training_quantization_ov(
        model: "CompiledModel",
        datamodule: AnomalibDataModule | None = None,
    ) -> "CompiledModel":
        """Apply post-training quantization to OpenVINO model.

        Args:
            model (CompiledModel): OpenVINO model to quantize
            datamodule (AnomalibDataModule | None): DataModule for calibration.
                Must contain at least 300 images. Defaults to ``None``

        Returns:
            CompiledModel: Quantized OpenVINO model

        Raises:
            ValueError: If datamodule is not provided
        """
        import nncf

        if datamodule is None:
            msg = "Datamodule must be provided for OpenVINO INT8_PTQ compression"
            raise ValueError(msg)
        datamodule.setup("fit")

        model_input = model.input(0)

        if model_input.partial_shape[0].is_static:
            datamodule.train_batch_size = model_input.shape[0]

        dataloader = datamodule.val_dataloader()
        if len(dataloader.dataset) < 300:
            logger.warning(
                f">300 images recommended for INT8 quantization, found only {len(dataloader.dataset)} images",
            )

        calibration_dataset = nncf.Dataset(dataloader, lambda x: x["image"])
        return nncf.quantize(model, calibration_dataset)

    @staticmethod
    def _accuracy_control_quantization_ov(
        model: "CompiledModel",
        datamodule: AnomalibDataModule | None = None,
        metric: Metric | None = None,
        task: TaskType | None = None,
    ) -> "CompiledModel":
        """Apply accuracy-aware quantization to OpenVINO model.

        Args:
            model (CompiledModel): OpenVINO model to quantize
            datamodule (AnomalibDataModule | None): DataModule for calibration
                and validation. Must contain at least 300 images.
                Defaults to ``None``
            metric (Metric | None): Metric to measure accuracy during quantization.
                Higher values should indicate better performance.
                Defaults to ``None``
            task (TaskType | None): Task type (classification/segmentation).
                Defaults to ``None``

        Returns:
            CompiledModel: Quantized OpenVINO model

        Raises:
            ValueError: If datamodule or metric is not provided
        """
        import nncf

        if datamodule is None:
            msg = "Datamodule must be provided for OpenVINO INT8_PTQ compression"
            raise ValueError(msg)
        datamodule.setup("fit")

        # if task is not provided, use the task from the datamodule
        task = task or datamodule.task

        if metric is None:
            msg = "Metric must be provided for OpenVINO INT8_ACQ compression"
            raise ValueError(msg)

        model_input = model.input(0)

        if model_input.partial_shape[0].is_static:
            datamodule.train_batch_size = model_input.shape[0]
            datamodule.eval_batch_size = model_input.shape[0]

        dataloader = datamodule.val_dataloader()
        if len(dataloader.dataset) < 300:
            logger.warning(
                f">300 images recommended for INT8 quantization, found only {len(dataloader.dataset)} images",
            )

        calibration_dataset = nncf.Dataset(dataloader, lambda x: x["image"])
        validation_dataset = nncf.Dataset(datamodule.test_dataloader())

        # validation function to evaluate the quality loss after quantization
        def val_fn(nncf_model: "CompiledModel", validation_data: Iterable) -> float:
            for batch in validation_data:
                preds = torch.from_numpy(nncf_model(batch["image"])[0])
                target = batch["label"] if task == TaskType.CLASSIFICATION else batch["mask"][:, None, :, :]
                metric.update(preds, target)
            return metric.compute()

        return nncf.quantize_with_accuracy_control(model, calibration_dataset, validation_dataset, val_fn)


def _create_export_root(export_root: str | Path, export_type: ExportType) -> Path:
    """Create directory structure for model export.

    Args:
        export_root (str | Path): Root directory for exports
        export_type (ExportType): Type of export (torch/onnx/openvino)

    Returns:
        Path: Created directory path
    """
    export_root = Path(export_root) / "weights" / export_type.value
    export_root.mkdir(parents=True, exist_ok=True)
    return export_root
