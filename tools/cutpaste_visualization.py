# Copyright (C) 2022-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Utilities to visualize CutPaste synthetic anomalies.

This module provides helper functions to generate PR-ready side-by-side examples:
``[Original | CutPaste | Mask]``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from PIL import Image
from torchvision.transforms import v2

from anomalib.data.utils.generators import CutPasteGenerator
from anomalib.data.utils.synthetic import DEFAULT_SYNTHETIC_MASK_THRESHOLD


def _to_display_image(image: torch.Tensor) -> np.ndarray:
    """Convert image tensor ``(C, H, W)`` to displayable numpy ``(H, W, C)``."""
    image = image.detach().cpu()
    if image.dtype != torch.float32:
        image = image.float()
    if image.max() > 1.0:
        image = image / 255.0
    image = image.clamp(0.0, 1.0)
    return image.permute(1, 2, 0).numpy()


def compute_cutpaste_mask(
    original_image: torch.Tensor,
    augmented_image: torch.Tensor,
    threshold: float = DEFAULT_SYNTHETIC_MASK_THRESHOLD,
) -> torch.Tensor:
    """Compute thresholded anomaly mask from image difference.

    Args:
        original_image (torch.Tensor): Original image tensor with shape ``(C, H, W)``.
        augmented_image (torch.Tensor): Augmented image tensor with shape ``(C, H, W)``.
        threshold (float): Difference threshold used for mask generation.
            Defaults to ``1e-3``.

    Returns:
        torch.Tensor: Binary mask with shape ``(1, H, W)``.
    """
    diff = (augmented_image - original_image).abs().sum(dim=0, keepdim=True)
    return (diff > threshold).float()


def visualize_cutpaste_example(
    image: torch.Tensor,
    generator: CutPasteGenerator,
    threshold: float = DEFAULT_SYNTHETIC_MASK_THRESHOLD,
    save_path: str | Path | None = None,
) -> Figure:
    """Create side-by-side visualization of CutPaste output.

    Args:
        image (torch.Tensor): Input image tensor with shape ``(C, H, W)``.
        generator (CutPasteGenerator): Configured CutPaste generator.
        threshold (float): Threshold used for difference-mask generation.
        save_path (str | Path | None): Optional output path to save PNG figure.

    Returns:
        Figure: Matplotlib figure containing ``[Original | CutPaste | Mask]``.
    """
    original_image = image.clone()
    augmented_image = generator.generate(image.clone())
    mask = compute_cutpaste_mask(original_image, augmented_image, threshold=threshold)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(_to_display_image(original_image))
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(_to_display_image(augmented_image))
    axes[1].set_title("CutPaste")
    axes[1].axis("off")

    axes[2].imshow(mask.squeeze(0).detach().cpu().numpy(), cmap="gray", vmin=0.0, vmax=1.0)
    axes[2].set_title("Mask")
    axes[2].axis("off")

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def load_image_as_tensor(
    image_path: str | Path,
    image_size: tuple[int, int] = (256, 256),
) -> torch.Tensor:
    """Load a real image from path and convert to tensor.

    Args:
        image_path (str | Path): Path to image file.
        image_size (tuple[int, int]): Resize target ``(H, W)``.

    Returns:
        torch.Tensor: Image tensor with shape ``(C, H, W)`` in ``[0, 1]``.
    """
    image = Image.open(image_path).convert("RGB")
    transform = v2.Compose([v2.Resize(image_size, antialias=True), v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    return transform(image)


def generate_examples(
    generator: CutPasteGenerator,
    num_samples: int = 3,
    save_dir: str | Path = "outputs",
    image_paths: Sequence[str | Path] | None = None,
    image_size: tuple[int, int] = (256, 256),
    threshold: float = DEFAULT_SYNTHETIC_MASK_THRESHOLD,
) -> list[Path]:
    """Generate and save multiple CutPaste examples for PR usage.

    If ``image_paths`` is provided, images are loaded from disk. Otherwise random
    synthetic RGB images are generated.

    Args:
        generator (CutPasteGenerator): Configured CutPaste generator.
        num_samples (int): Number of examples to save.
        save_dir (str | Path): Directory where ``example_*.png`` are written.
        image_paths (Sequence[str | Path] | None): Optional input image paths.
        image_size (tuple[int, int]): Target image size for loaded/random images.
        threshold (float): Threshold used for mask generation.

    Returns:
        list[Path]: Paths of generated PNG files.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for idx in range(num_samples):
        if image_paths:
            image = load_image_as_tensor(image_paths[idx % len(image_paths)], image_size=image_size)
        else:
            image = torch.rand(3, image_size[0], image_size[1])

        out_path = save_dir / f"example_{idx + 1}.png"
        fig = visualize_cutpaste_example(
            image=image,
            generator=generator,
            threshold=threshold,
            save_path=out_path,
        )
        plt.close(fig)
        saved_paths.append(out_path)

    return saved_paths


def _collect_image_paths(input_path: str | Path | None) -> list[Path] | None:
    """Collect image paths from file/directory input."""
    if input_path is None:
        return None
    input_path = Path(input_path)
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
        return sorted([path for path in input_path.rglob("*") if path.suffix.lower() in exts])
    msg = f"Input path does not exist: {input_path}"
    raise FileNotFoundError(msg)


def main() -> None:
    """CLI entrypoint to generate CutPaste visualization examples."""
    parser = argparse.ArgumentParser(description="Generate CutPaste visualization examples.")
    parser.add_argument("--input", type=str, default=None, help="Image file or directory. If omitted, random images are used.")
    parser.add_argument("--output-dir", type=str, default="outputs/cutpaste_examples", help="Directory to save example PNG files.")
    parser.add_argument("--num-samples", type=int, default=3, help="Number of examples to generate.")
    parser.add_argument("--height", type=int, default=256, help="Image height.")
    parser.add_argument("--width", type=int, default=256, help="Image width.")
    parser.add_argument("--probability", type=float, default=1.0, help="CutPaste application probability.")
    parser.add_argument("--blend-factor", type=float, default=0.5, help="CutPaste blend factor.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["normal", "scar", "union"],
        default="normal",
        help="CutPaste variant to visualize.",
    )
    parser.add_argument(
        "--rotation-min",
        type=float,
        default=0.0,
        help="Minimum CutPaste rotation angle. Use 0 for artifact-free PR previews.",
    )
    parser.add_argument(
        "--rotation-max",
        type=float,
        default=0.0,
        help="Maximum CutPaste rotation angle. Use 0 for artifact-free PR previews.",
    )
    parser.add_argument("--threshold", type=float, default=DEFAULT_SYNTHETIC_MASK_THRESHOLD, help="Mask threshold.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed.")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    generator = CutPasteGenerator(
        probability=args.probability,
        mode=args.mode,
        blend_factor=args.blend_factor,
        rotation_range=(args.rotation_min, args.rotation_max),
    )
    image_paths = _collect_image_paths(args.input)
    if image_paths is not None and not image_paths:
        msg = f"No supported images found under: {args.input}"
        raise ValueError(msg)

    saved_paths = generate_examples(
        generator=generator,
        num_samples=args.num_samples,
        save_dir=args.output_dir,
        image_paths=image_paths,
        image_size=(args.height, args.width),
        threshold=args.threshold,
    )
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
