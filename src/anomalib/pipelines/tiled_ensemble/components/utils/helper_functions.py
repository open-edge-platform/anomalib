"""Helper functions for the tiled ensemble training."""

# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from pathlib import Path

from jsonargparse import ArgumentParser, Namespace
from lightning import Trainer
from torchvision.transforms.v2 import Compose, Resize, Transform

from anomalib.data import AnomalibDataModule, ImageBatch, get_datamodule
from anomalib.models import AnomalibModule, get_model
from anomalib.utils.normalization import NormalizationMethod

from . import NormalizationStage
from .ensemble_engine import TiledEnsembleEngine
from .ensemble_tiling import EnsembleTiler, TileCollater

logger = logging.getLogger(__name__)


def get_ensemble_datamodule(config: dict, tiler: EnsembleTiler, tile_index: tuple[int, int]) -> AnomalibDataModule:
    """Get Anomaly Datamodule adjusted for use in ensemble.

    Datamodule collate function gets replaced by TileCollater in order to tile all images before they are passed on.

    Args:
        config: tiled ensemble data configuration.
        tiler (EnsembleTiler): Tiler used to split the images to tiles for use in ensemble.
        tile_index (tuple[int, int]): Index of the tile in the split image.

    Returns:
        AnomalibDataModule: Anomalib Lightning DataModule
    """
    datamodule = get_datamodule(config)

    # add tiled ensemble image_size transform to datamodule
    setup_transforms(datamodule, config)
    datamodule.external_collate_fn = TileCollater(tiler, tile_index, default_collate_fn=ImageBatch.collate)
    datamodule.setup()

    return datamodule


def setup_transforms(datamodule: AnomalibDataModule, config: dict) -> None:
    """Modify datamodule resize transforms so the effective ensemble image_size is correct.

    Args:
        datamodule: datamodule where resize transform will be setup.
        config: tiled ensemble config

    """
    tiled_ens_size = config["tiling"]["image_size"]
    resize_transform = Resize(tiled_ens_size)

    for subset_name in ["train", "val", "test"]:
        default_aug = getattr(datamodule, f"{subset_name}_augmentations", None)

        if isinstance(default_aug, Resize):
            msg = f"Conflicting resize shapes found between dataset augmentations and tiled ensemble size. \
                You are using a Resize transform in your input data augmentations. Please be aware that the \
                tiled ensemble image size is determined by tiling config. The final effective input size as \
                seen by individual model will be determined by the tile_size. To change \
                the effective ensemble input size, please change the image_size in the tiling config. \
                Augmentations: {default_aug.size}, Tiled ensemble base size: {tiled_ens_size}"
            logger.warning(msg)
            augmentations = resize_transform
        elif isinstance(default_aug, Compose):
            augmentations = Compose([*default_aug.transforms, resize_transform])
        elif isinstance(default_aug, Transform):
            augmentations = Compose([default_aug, resize_transform])
        else:
            augmentations = resize_transform
        # add augmentations with resize to datamodule, ensuring that output images match effective size
        setattr(datamodule, f"{subset_name}_augmentations", augmentations)


def get_ensemble_model(model_args: dict, tiler: EnsembleTiler) -> AnomalibModule:
    """Get model prepared for ensemble training.

    Args:
        model_args: tiled ensemble model configuration.
        tiler (EnsembleTiler): tiler used to get tile dimensions.

    Returns:
        AnomalyModule: model with input_size setup
    """
    model = get_model(model_args)
    # set model input size match tile size
    model.set_input_size((tiler.tile_size_h, tiler.tile_size_w))

    return model


def get_ensemble_tiler(tiling_args: dict) -> EnsembleTiler:
    """Get tiler used for image tiling and to obtain tile dimensions.

    Args:
        tiling_args: tiled ensemble tiling configuration.

    Returns:
        EnsembleTiler: tiler object.
    """
    tiler = EnsembleTiler(
        tile_size=tiling_args["tile_size"],
        stride=tiling_args["stride"],
        image_size=tiling_args["image_size"],
    )

    return tiler  # noqa: RET504


def parse_trainer_kwargs(trainer_args: dict | None) -> Namespace | dict:
    """Parse trainer args and instantiate all needed elements.

    Transforms config into kwargs ready for Trainer, including instantiation of callback etc.

    Args:
        trainer_args (dict): Trainer args dictionary.

    Returns:
        dict: parsed kwargs with instantiated elements.
    """
    if not trainer_args:
        return {}

    # try to get trainer args, if not present return empty
    parser = ArgumentParser()

    parser.add_class_arguments(Trainer, fail_untyped=False, instantiate=False, sub_configs=True)
    config = parser.parse_object(trainer_args)
    objects = parser.instantiate_classes(config)

    return objects  # noqa: RET504


def get_ensemble_engine(
    tile_index: tuple[int, int],
    accelerator: str,
    devices: list[int] | str | int,
    root_dir: Path,
    normalization_stage: str,
    metrics: dict | None = None,
    trainer_args: dict | None = None,
) -> TiledEnsembleEngine:
    """Prepare engine for ensemble training or prediction.

    This method makes sure correct normalization is used, prepares metrics and additional trainer kwargs..

    Args:
        tile_index (tuple[int, int]): Index of tile that this model processes.
        accelerator (str): Accelerator (device) to use.
        devices (list[int] | str | int): device IDs used for training.
        root_dir (Path): Root directory to save checkpoints, stats and images.
        normalization_stage (str): Config dictionary for ensemble post-processing.
        metrics (dict): Dict containing pixel and image metrics names.
        trainer_args (dict): Trainer args dictionary. Empty dict if not present.

    Returns:
        TiledEnsembleEngine: set up engine for ensemble training/prediction.
    """
    # if we want tile level normalization we set it here, otherwise it's done later on joined images
    if normalization_stage == NormalizationStage.TILE:
        normalization = NormalizationMethod.MIN_MAX
    else:
        normalization = NormalizationMethod.NONE

    # parse additional trainer args and callbacks if present in config
    trainer_kwargs = parse_trainer_kwargs(trainer_args)
    # remove keys that we already have
    trainer_kwargs.pop("accelerator", None)
    trainer_kwargs.pop("default_root_dir", None)
    trainer_kwargs.pop("devices", None)

    # create engine for specific tile location
    engine = TiledEnsembleEngine(
        tile_index=tile_index,
        normalization=normalization,
        accelerator=accelerator,
        devices=devices,
        default_root_dir=root_dir,
        image_metrics=metrics.get("image", None) if metrics else None,
        pixel_metrics=metrics.get("pixel", None) if metrics else None,
        **trainer_kwargs,
    )

    return engine  # noqa: RET504


def get_threshold_values(normalization_stage: NormalizationStage, root_dir: Path) -> tuple[float, float]:
    """Get threshold values for image and pixel level predictions.

    If normalization is not used, get values based on statistics obtained from validation set.
    If normalization is used, both image and pixel threshold are 0.5

    Args:
        normalization_stage (NormalizationStage): ensemble run args, used to get normalization stage.
        root_dir (Path): path to run root where stats file is saved.

    Returns:
        tuple[float, float]: image and pixel threshold.
    """
    if normalization_stage == NormalizationStage.NONE:
        stats_path = root_dir / "weights" / "lightning" / "stats.json"
        with stats_path.open("r") as f:
            stats = json.load(f)
        image_threshold = stats["image_threshold"]
        pixel_threshold = stats["pixel_threshold"]
    else:
        # normalization transforms the scores so that threshold is at 0.5
        image_threshold = 0.5
        pixel_threshold = 0.5

    return image_threshold, pixel_threshold
