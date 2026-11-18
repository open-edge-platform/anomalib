# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pretrain UniADet's auxiliary head with Lightning, outside Engine's target-domain lifecycle.

Example:
    uv run python -m tools.uniadet.train --dataset visa --root datasets/visa/visa_pytorch --output results/UniADet/visa
"""

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path

import lightning
import timm
import torch
from lightning.pytorch import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from anomalib.data import ImageBatch
from anomalib.models import UniADet
from anomalib.models.image.uniadet.loss import UniADetLoss

from .data import AuxiliaryDataset

logger = logging.getLogger(__name__)


class AuxiliaryTrainingModule(LightningModule):
    """Train only the decoupled head on labeled auxiliary images.

    Args:
        detector (UniADet): Detector with a frozen pretrained encoder.
        learning_rate (float): Adam learning rate, default ``0.001``.
    """

    def __init__(self, detector: UniADet, learning_rate: float = 0.001) -> None:
        super().__init__()
        self.detector = detector
        self.loss = UniADetLoss()
        self.learning_rate = learning_rate

    def training_step(self, batch: ImageBatch, batch_idx: int) -> torch.Tensor:
        """Optimize per-layer image cross-entropy and pixel focal/Dice objectives.

        Args:
            batch (ImageBatch): Auxiliary images with real labels and masks.
            batch_idx (int): Batch index (unused).

        Returns:
            torch.Tensor: Differentiable scalar objective.
        """
        del batch_idx
        if self.detector.pre_processor is None:
            msg = "Auxiliary training requires backbone preprocessing."
            raise ValueError(msg)
        images = self.detector.pre_processor(batch.image)
        model = self.detector.model
        features = model.backbone(images)
        logits = model.head(features, model.backbone.grid_size)
        loss = self.loss(logits, batch.gt_label, batch.gt_mask)
        self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=len(images))
        return loss

    def configure_optimizers(self) -> torch.optim.Adam:
        """Return the paper recipe's head-only Adam optimizer."""
        return torch.optim.Adam(self.detector.model.head.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))

    def on_train_epoch_end(self) -> None:
        """Report completed epochs and the observed loss for long reproduction runs."""
        logger.info("Epoch %d: loss %.6f", self.current_epoch + 1, self.trainer.callback_metrics["train_loss"])


def main() -> None:
    """Run reproducible auxiliary training and save head-only weights plus a provenance manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("mvtec", "visa"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--backbone", choices=("clip", "dinov2", "dinov3"), default="dinov2")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=-1, help="Smoke-run limit; not a paper reproduction.")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    seed_everything(args.seed, workers=True)
    torch.set_num_threads(4)
    detector = UniADet(backbone=args.backbone, post_processor=False, evaluator=False, visualizer=False)
    dataset = AuxiliaryDataset(args.root, args.dataset, detector.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=ImageBatch.collate,
        pin_memory=True,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = ModelCheckpoint(dirpath=args.output / "resume", save_last=True, save_top_k=0, every_n_epochs=1)
    trainer = Trainer(
        accelerator="gpu",
        devices=[args.device],
        max_epochs=args.epochs,
        max_steps=args.max_steps,
        precision="32-true",
        deterministic=True,
        logger=False,
        callbacks=[checkpoint],
        enable_progress_bar=False,
        default_root_dir=args.output,
    )
    module = AuxiliaryTrainingModule(detector)
    start = time.perf_counter()
    trainer.fit(module, train_dataloaders=loader, ckpt_path=args.resume)
    detector.model.head_trained.fill_(value=True)
    head_path = args.output / "head.pt"
    detector.model.save_head(head_path)
    manifest = {
        "implementation": "independent-paper",
        "recipe_version": 2,
        "source_dataset": args.dataset,
        "source_split": "test",
        "backbone": args.backbone,
        "layers": list(detector.model.layers),
        "image_size": detector.image_size,
        "epochs": args.epochs,
        "completed_steps": trainer.global_step,
        "samples": len(dataset),
        "batch_size": args.batch_size,
        "seed": args.seed,
        "optimizer": {"name": "Adam", "lr": 0.001, "betas": [0.5, 0.999]},
        "caa": {"probability": 0.5, "grids": [2, 3], "mosaic_probability": 0.5},
        "augmentation": {
            "library": "torchvision",
            "color_jitter": {"brightness": 0.4, "contrast": 0.4, "saturation": 0.4, "hue": 0.1, "probability": 0.5},
            "rotate90_probability": 0.5,
            "channel_shuffle_probability": 0.25,
            "grayscale_probability": 0.25,
            "horizontal_flip_probability": 0.5,
            "vertical_flip_probability": 0.5,
        },
        "temperature": detector.model.head.temperature,
        "prototype_initialization_std": 0.02,
        "loss": {
            "focal_gamma": 2.0,
            "focal_smoothing": 1e-5,
            "dice_smoothing": 1.0,
            "dice_denominator": "linear",
            "resize_probabilities": True,
        },
        "duration_seconds": time.perf_counter() - start,
        "complete_recipe": args.epochs == 15 and args.batch_size == 16 and args.max_steps == -1 and args.seed == 10,
        "head_sha256": hashlib.sha256(head_path.read_bytes()).hexdigest(),
        "torch": torch.__version__,
        "timm": timm.__version__,
        "lightning": lightning.__version__,
        "gpu": torch.cuda.get_device_name(args.device),
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(args.device),
    }
    (args.output / "training.json").write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info("Saved auxiliary head to %s", head_path)


if __name__ == "__main__":
    main()
