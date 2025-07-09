# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Anomalib Gradio Script.

This script provide a gradio web interface
"""

from argparse import ArgumentParser
from importlib import import_module
from pathlib import Path

from lightning_utilities.core.imports import module_available
from PIL.Image import Image

from anomalib.deploy import OpenVINOInferencer, TorchInferencer
from anomalib.visualization.image.functional import overlay_image, visualize_anomaly_map, visualize_mask

if not module_available("gradio"):
    msg = "Gradio is not installed. Please install it using: pip install gradio"
    raise ImportError(msg)

import gradio


def get_parser() -> ArgumentParser:
    """Get command line arguments.

    Example:
        Example for Torch Inference.
        >>> python tools/inference/gradio_inference.py  \
        ...     --weights ./results/padim/mvtecad/bottle/weights/torch/model.pt

    Returns:
        ArgumentParser: Argument parser for gradio inference.
    """
    parser = ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True, help="Path to model weights")
    parser.add_argument("--share", type=bool, required=False, default=False, help="Share Gradio `share_url`")

    return parser


def get_inferencer(weight_path: Path) -> OpenVINOInferencer | TorchInferencer:
    """Parse args and open inferencer.

    Args:
        weight_path (Path): Path to model weights.
        metadata (Path | None, optional): Metadata is required for OpenVINO models. Defaults to None.

    Raises:
        ValueError: If unsupported model weight is passed.

    Returns:
        Inferencer: Torch or OpenVINO inferencer.
    """
    # Get the inferencer. We use .ckpt extension for Torch models and (onnx, bin)
    # for the openvino models.
    extension = weight_path.suffix
    inferencer: OpenVINOInferencer | TorchInferencer
    module = import_module("anomalib.deploy")
    if extension in {".pt", ".pth", ".ckpt"}:
        torch_inferencer = module.TorchInferencer
        inferencer = torch_inferencer(path=weight_path)

    elif extension in {".onnx", ".bin", ".xml"}:
        openvino_inferencer = module.OpenVINOInferencer
        inferencer = openvino_inferencer(path=weight_path)

    else:
        msg = (
            "Model extension is not supported. "
            "Torch Inferencer exptects a .ckpt file,OpenVINO Inferencer expects either .onnx, .bin or .xml file. "
            f"Got {extension}"
        )
        raise ValueError(
            msg,
        )

    return inferencer


def infer(
    image: Image,
    inferencer: OpenVINOInferencer | TorchInferencer,
) -> tuple[Image, Image]:
    """Inference function, return anomaly map, score, heat map, prediction mask ans visualisation.

    Args:
        image (Image): image to compute
        inferencer (OpenVINOInferencer | TorchInferencer): model inferencer

    Returns:
        tuple[Image, Image]: heat_map, segmentation result.
    """
    # Perform inference for the given image.
    predictions = inferencer.predict(image=image)

    # Create visualization of the anomaly map on the input image.
    anomaly_map = visualize_anomaly_map(predictions.anomaly_map)
    heat_map = overlay_image(base=image, overlay=anomaly_map)

    # Create visualization of the predicted mask on the input image.
    pred_mask = visualize_mask(predictions.pred_mask, mode="contour")
    segmentation = overlay_image(base=image, overlay=pred_mask)

    return (heat_map, segmentation)


if __name__ == "__main__":
    args = get_parser().parse_args()
    gradio_inferencer = get_inferencer(args.weights)

    interface = gradio.Interface(
        fn=lambda image: infer(image, gradio_inferencer),
        inputs=gradio.Image(
            image_mode="RGB",
            sources=["upload", "webcam"],
            type="pil",
            label="Image",
        ),
        outputs=[
            gradio.Image(type="pil", label="Predicted Heat Map"),
            gradio.Image(type="pil", label="Segmentation Result"),
        ],
        title="Anomalib",
        description="Anomalib Gradio",
    )

    interface.launch(share=args.share)
