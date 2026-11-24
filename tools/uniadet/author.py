# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Import Apache-2.0 UniADet heads using an authorized Hugging Face login."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from anomalib.models.image.uniadet.backbones import BACKBONES

REPOSITORY = "csgaobb/UniADet"
REVISION = "88de3178cff5ab6f895f53d5934dac9f3f80b3b8"
SOURCE_REVISION = "2b81369d418ed08fd882452597ad5ebe7df3cb1d"
LICENSE_URL = "https://github.com/gaobb/UniADet/blob/45272f0e65ba4e27f4921cd4c9819c40dd8e5aa4/LICENSE"
LICENSE_CONFIRMATION = "https://github.com/gaobb/UniADet/issues/7#issuecomment-7075731743"
AUTHOR_NAMES = {
    "clip": "ViT-L-14@336px",
    "dinov2": "ViT-L-14-DINOv2Reg",
    "dinov3": "ViT-L-16-DINOv3",
}


def convert_head(checkpoint: dict[str, object], feature_dim: int) -> dict[str, torch.Tensor]:
    """Validate and stack the ten author prototype tensors into the integrated head layout.

    Args:
        checkpoint (dict[str, object]): Tensor-only checkpoint loaded with ``weights_only=True``.
        feature_dim (int): Expected token dimension for the explicitly selected backbone.

    Returns:
        dict[str, torch.Tensor]: Image and pixel prototype tensors, preserving every value.

    Raises:
        ValueError: If keys, shapes, tensor values or types do not match the released head.
    """
    state = checkpoint.get("msdcs")
    expected = {f"{kind}_prompts.{index}.weight" for kind in ("class", "patch") for index in range(5)}
    if not isinstance(state, dict) or set(state) != expected:
        msg = "Expected exactly five class and five patch prototype tensors under 'msdcs'."
        raise ValueError(msg)
    if any(
        not isinstance(value, torch.Tensor)
        or value.shape != (2, feature_dim)
        or not value.is_floating_point()
        or not torch.isfinite(value).all()
        or (value.norm(dim=-1) == 0).any()
        for value in state.values()
    ):
        msg = "Author prototypes must be finite, nonzero floating tensors with the selected backbone dimension."
        raise ValueError(msg)
    return {
        target: torch.stack([state[f"{source}_prompts.{index}.weight"] for index in range(5)])
        for target, source in (("image_prototypes", "class"), ("pixel_prototypes", "patch"))
    }


def download_head(backbone: str, source_dataset: str, output: Path) -> Path:
    """Download and convert one authorized author head with pinned provenance.

    Args:
        backbone (str): ``"clip"``, ``"dinov2"`` or ``"dinov3"``.
        source_dataset (str): Auxiliary training dataset, ``"mvtec"`` or ``"visa"``.
        output (Path): Local-only destination; model binaries must not be committed.

    Returns:
        Path: Converted checkpoint with explicit author-protocol configuration.

    Raises:
        ValueError: If the backbone or dataset is unsupported or the checkpoint is malformed.
    """
    if backbone not in BACKBONES or source_dataset not in {"mvtec", "visa"}:
        msg = "Select a supported backbone and auxiliary dataset."
        raise ValueError(msg)
    spec = BACKBONES[backbone]
    filename = f"train_on_{source_dataset}_zs_{AUTHOR_NAMES[backbone]}_imagesize{spec.image_size}_1015/epoch_15.pth"
    original = Path(hf_hub_download(REPOSITORY, filename, revision=REVISION))
    # Keep the associated model card and terms in the authorized local cache too.
    hf_hub_download(REPOSITORY, "README.md", revision=REVISION)
    hf_hub_download(REPOSITORY, "config.json", revision=REVISION)
    tensors = convert_head(torch.load(original, map_location="cpu", weights_only=True), spec.feature_dim)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "head.pt"
    torch.save(
        {
            "config": {
                "backbone": backbone,
                "layers": [12, 15, 18, 21, 24],
                "image_size": spec.image_size,
                "temperature": 0.07,
                "protocol": "author",
            },
            "state_dict": tensors,
        },
        destination,
    )
    provenance = {
        "origin": "author",
        "backbone": backbone,
        "source_dataset": source_dataset,
        "repository": REPOSITORY,
        "revision": REVISION,
        "filename": filename,
        "original_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
        "head_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "conversion": "Stack only; no tensor values changed.",
        "source_revision": SOURCE_REVISION,
        "protocol": "author-v1",
        "training_reproduced": False,
        "license": "Apache-2.0",
        "license_url": LICENSE_URL,
        "license_confirmation": LICENSE_CONFIRMATION,
        "access_terms": "Gated Hugging Face download; see the pinned model card.",
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return destination


def main() -> None:
    """Download selected heads or the complete reproduction matrix."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backbone", choices=tuple(BACKBONES))
    parser.add_argument("--source-dataset", choices=("mvtec", "visa"))
    args = parser.parse_args()
    backbones = (args.backbone,) if args.backbone else tuple(BACKBONES)
    datasets = (args.source_dataset,) if args.source_dataset else ("mvtec", "visa")
    for backbone in backbones:
        for dataset in datasets:
            download_head(backbone, dataset, args.output / f"{backbone}_{dataset}")


if __name__ == "__main__":
    main()
