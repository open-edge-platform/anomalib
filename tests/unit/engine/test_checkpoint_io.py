# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for AnomalibCheckpointIO and Engine plugin installation."""

from enum import Enum
from pathlib import Path

import torch
from lightning.fabric.plugins.io.torch_io import TorchCheckpointIO
from lightning.pytorch import Trainer
from torchvision.transforms.v2 import Resize

from anomalib import PrecisionType
from anomalib.engine import Engine
from anomalib.engine.plugins import AnomalibCheckpointIO
from anomalib.models import Padim
from anomalib.models.components.base import AnomalibModule
from anomalib.models.image.efficient_ad import EfficientAd
from anomalib.models.image.efficient_ad.torch_model import EfficientAdModelSize
from anomalib.models.image.vlm_ad import VlmAd
from anomalib.models.image.vlm_ad.utils import ModelName
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.pre_processing.utils.spec import transform_to_spec


class _ExtraEnum(Enum):
    """Local enum used only to exercise CheckpointIO extras."""

    VALUE = "value"


class _ModelWithExtraGlobals(Padim):
    """Padim subclass that declares an extra safe global for tests."""

    @classmethod
    def checkpoint_safe_globals(cls) -> tuple[type[Enum], ...]:
        return (_ExtraEnum,)


def test_checkpoint_io_loads_precision_type(tmp_path: Path) -> None:
    """AnomalibCheckpointIO can weights_only-load PrecisionType hyperparameters."""
    path = tmp_path / "hparams.pt"
    # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
    torch.save({"precision": PrecisionType.FLOAT32}, path)

    loaded = AnomalibCheckpointIO().load_checkpoint(path, weights_only=True)

    assert loaded["precision"] == PrecisionType.FLOAT32


def test_checkpoint_io_loads_extra_safe_globals(tmp_path: Path) -> None:
    """AnomalibCheckpointIO merges constructor extras into the allowlist."""
    path = tmp_path / "hparams.pt"
    # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
    torch.save({"extra": _ExtraEnum.VALUE}, path)

    loaded = AnomalibCheckpointIO(extra_safe_globals=[_ExtraEnum]).load_checkpoint(
        path,
        weights_only=True,
    )

    assert loaded["extra"] == _ExtraEnum.VALUE


def test_engine_installs_anomalib_checkpoint_io(tmp_path: Path) -> None:
    """Engine installs AnomalibCheckpointIO when the user does not pass a CheckpointIO."""
    engine = Engine(default_root_dir=tmp_path, logger=False)
    engine._setup_trainer(Padim())  # noqa: SLF001

    assert isinstance(engine.trainer.strategy.checkpoint_io, AnomalibCheckpointIO)


def test_engine_sets_checkpoint_io_extras_from_model(tmp_path: Path) -> None:
    """Engine copies ``checkpoint_safe_globals`` onto AnomalibCheckpointIO."""
    engine = Engine(default_root_dir=tmp_path, logger=False)
    engine._setup_trainer(_ModelWithExtraGlobals())  # noqa: SLF001

    checkpoint_io = engine.trainer.strategy.checkpoint_io
    assert isinstance(checkpoint_io, AnomalibCheckpointIO)
    assert checkpoint_io.extra_safe_globals == [_ExtraEnum]


def test_engine_preserves_user_checkpoint_io(tmp_path: Path) -> None:
    """Engine leaves a user-supplied CheckpointIO in place."""
    user_io = TorchCheckpointIO()
    engine = Engine(default_root_dir=tmp_path, logger=False, plugins=[user_io])
    engine._setup_trainer(Padim())  # noqa: SLF001

    assert engine.trainer.strategy.checkpoint_io is user_io
    assert not isinstance(engine.trainer.strategy.checkpoint_io, AnomalibCheckpointIO)


def test_vlmad_declares_model_name_safe_global() -> None:
    """VlmAd allowlists ModelName for weights_only checkpoint restores."""
    assert ModelName in VlmAd.checkpoint_safe_globals()
    assert AnomalibModule.checkpoint_safe_globals() == ()


def test_efficient_ad_declares_model_size_safe_global() -> None:
    """EfficientAd allowlists EfficientAdModelSize for weights_only restores."""
    assert EfficientAdModelSize in EfficientAd.checkpoint_safe_globals()


def test_custom_preprocessor_round_trips_as_plain_data(tmp_path: Path) -> None:
    """Checkpoint restore preserves custom preprocessing without pickled modules."""
    transform = Resize((128, 192), antialias=False)
    model = Padim(pre_processor=PreProcessor(transform=transform))
    trainer = Trainer(max_epochs=1, logger=False, barebones=True)
    trainer.strategy.connect(model)
    checkpoint_path = tmp_path / "padim.ckpt"
    trainer.save_checkpoint(checkpoint_path)

    checkpoint = AnomalibCheckpointIO().load_checkpoint(checkpoint_path, weights_only=True)
    loaded = Padim.load_from_checkpoint(checkpoint_path, weights_only=True)

    assert checkpoint["anomalib_pre_processor_spec"] == transform_to_spec(transform)
    assert transform_to_spec(loaded.pre_processor.transform) == transform_to_spec(transform)


def test_path_hyperparameters_are_saved_as_strings(tmp_path: Path) -> None:
    """Path hyperparameters load safely and are reconstructed by constructors."""
    from anomalib.models import EfficientAd

    model = EfficientAd(imagenet_dir=tmp_path / "imagenette")
    trainer = Trainer(max_epochs=1, logger=False, barebones=True)
    trainer.strategy.connect(model)
    checkpoint_path = tmp_path / "efficientad.ckpt"
    trainer.save_checkpoint(checkpoint_path)

    loaded_checkpoint = AnomalibCheckpointIO(
        extra_safe_globals=EfficientAd.checkpoint_safe_globals(),
    ).load_checkpoint(checkpoint_path, weights_only=True)
    loaded = EfficientAd.load_from_checkpoint(checkpoint_path, weights_only=True)

    assert loaded_checkpoint["hyper_parameters"]["imagenet_dir"] == str(tmp_path / "imagenette")
    assert loaded.imagenet_dir == tmp_path / "imagenette"


def test_postprocessor_config_round_trips(tmp_path: Path) -> None:
    """Postprocessor inference configuration survives checkpoint restore."""
    processor = PostProcessor(
        enable_normalization=False,
        enable_thresholding=False,
        enable_threshold_matching=False,
        image_sensitivity=0.7,
        pixel_sensitivity=0.3,
    )
    model = Padim(post_processor=processor)
    trainer = Trainer(max_epochs=1, logger=False, barebones=True)
    trainer.strategy.connect(model)
    checkpoint_path = tmp_path / "postprocessor.ckpt"
    trainer.save_checkpoint(checkpoint_path)

    loaded = Padim.load_from_checkpoint(checkpoint_path, weights_only=True)

    assert loaded.post_processor.enable_normalization is False
    assert loaded.post_processor.enable_thresholding is False
    assert loaded.post_processor.enable_threshold_matching is False
    assert loaded.post_processor.image_sensitivity == 0.7
    assert loaded.post_processor.pixel_sensitivity == 0.3
