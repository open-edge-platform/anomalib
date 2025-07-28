# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Anomalib OpenVINO Inferencer Script.

This script performs OpenVINO inference by reading a model from
file system, and show the visualization results.
"""

import logging
import warnings
from argparse import ArgumentParser, Namespace
from pathlib import Path

from tqdm import tqdm

from anomalib.data.utils import generate_output_image_filename, get_image_filenames
from anomalib.deploy import OpenVINOInferencer
from anomalib.visualization import visualize_image_item

logger = logging.getLogger(__name__)


def get_parser() -> ArgumentParser:
    """Get parser.

    Returns:
        ArgumentParser: The parser object.
    """
    parser = ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True, help="Path to model weights")
    parser.add_argument("--input", type=Path, required=True, help="Path to an image to infer.")
    parser.add_argument("--output", type=Path, required=False, help="Path to save the output image.")
    parser.add_argument(
        "--task",
        type=str,
        required=False,
        help="Task type.",
        default="classification",
        choices=["classification", "detection", "segmentation"],
    )
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        help="Hardware device on which the model will be deployed",
        default="CPU",
        choices=["CPU", "GPU", "VPU"],
    )
    parser.add_argument(
        "--visualization_mode",
        type=str,
        required=False,
        default="simple",
        help="Visualization mode.",
        choices=["full", "simple"],
    )
    parser.add_argument(
        "--show",
        action="store_true",
        required=False,
        help="Show the visualized predictions on the screen.",
    )

    return parser


def infer(args: Namespace) -> None:
    """Infer predictions.

    Show/save the output if path is to an image. If the path is a directory, go over each image in the directory.

    Args:
        args (Namespace): The arguments from the command line.
    """
    # Get the inferencer.
    inferencer = OpenVINOInferencer(path=args.weights, device=args.device)

    filenames = get_image_filenames(path=args.input)
    for filename in tqdm(filenames, desc="Predicting images"):
        predictions = inferencer.predict(filename)

        # NOTE: This visualization approach is experimental and might change in the future.
        output = visualize_image_item(
            item=predictions.items[0],
            fields=["image"],  # Can be used to visualize other fields such as anomaly_map, pred_mask, etc.
            overlay_fields=[
                # Can be used to overlay multiple other fields.
                ("image", ["anomaly_map"]),
                ("image", ["pred_mask"]),
            ],
        )

        if args.output is None and args.show is False:
            msg = "Neither output path is provided nor show flag is set. Inferencer will run but return nothing."
            logger.warning(msg)

        if output is not None:
            if args.output:
                file_path = generate_output_image_filename(input_path=filename, output_path=args.output)
                output.save(file_path)

            # Show the image in case the flag is set by the user.
            if args.show:
                output.show("Output Image")


if __name__ == "__main__":
    args = get_parser().parse_args()

    # Deprecation warning for --task argument
    if hasattr(args, "task"):
        warnings.warn(
            "The --task argument is deprecated and no longer used. It will be removed in a future release.",
            FutureWarning,
            stacklevel=2,
        )

    # Deprecation warning for --visualization_mode argument
    if hasattr(args, "visualization_mode"):
        warnings.warn(
            "The --visualization_mode argument is deprecated and no longer used. "
            "It will be removed in a future release.",
            FutureWarning,
            stacklevel=2,
        )

    infer(args)
