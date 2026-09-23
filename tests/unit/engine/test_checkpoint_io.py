# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for AnomalibCheckpointIO and Engine plugin installation."""

from enum import Enum
from pathlib import Path

import pytest
import torch
from lightning.fabric.plugins.io.torch_io import TorchCheckpointIO
from lightning.pytorch import Trainer
from torch import nn
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
from anomalib.post_processing import MEBinPostProcessor, PostProcessor
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

    assert checkpoint["anomalib_pre_processor_config"] == {"transform": transform_to_spec(transform)}
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


def test_mebin_post_processor_config_round_trips(tmp_path: Path) -> None:
    """MEBinPostProcessor's extra configuration survives checkpoint restore.

    MEBinPostProcessor only extends PostProcessor's ``_checkpoint_config_keys``
    property rather than overriding ``checkpoint_config``/``load_checkpoint_config``,
    so this also verifies that the base hooks pick up the extension.

    ``load_from_checkpoint`` reconstructs ``Padim`` from its saved
    hyperparameters, and ``post_processor`` is intentionally excluded from
    those (see ``AnomalibModule.__init__``), so the default
    ``configure_post_processor`` would otherwise build a plain
    ``PostProcessor``. As with a custom pre-processor, the caller must supply
    a fresh instance of the same custom type at load time; only its
    *configuration* is restored from the checkpoint, not its class.
    """
    processor = MEBinPostProcessor(sample_rate=8, min_interval_len=2, erode=False, kernel_size=3)
    model = Padim(post_processor=processor)
    trainer = Trainer(max_epochs=1, logger=False, barebones=True)
    trainer.strategy.connect(model)
    checkpoint_path = tmp_path / "mebin.ckpt"
    trainer.save_checkpoint(checkpoint_path)

    loaded = Padim.load_from_checkpoint(
        checkpoint_path,
        weights_only=True,
        post_processor=MEBinPostProcessor(),
    )

    assert isinstance(loaded.post_processor, MEBinPostProcessor)
    assert loaded.post_processor.sample_rate == 8
    assert loaded.post_processor.min_interval_len == 2
    assert loaded.post_processor.erode is False
    assert loaded.post_processor.kernel_size == 3


class _BareModulePreProcessor(nn.Module):
    """A pre-processor that only satisfies the ``nn.Module`` type hint.

    Simulates a user who passes a bare ``nn.Module`` as ``pre_processor``,
    which the public API allows but which implements neither
    ``checkpoint_config`` nor ``.transform``.
    """


class _BareModulePostProcessor(nn.Module):
    """A post-processor that only satisfies the ``nn.Module`` type hint."""


@pytest.mark.parametrize(
    ("kwarg", "component_cls", "checkpoint_key"),
    [
        ("pre_processor", _BareModulePreProcessor, "anomalib_pre_processor_config"),
        ("post_processor", _BareModulePostProcessor, "anomalib_post_processor_config"),
    ],
)
def test_bare_module_component_is_skipped_not_raised(
    tmp_path: Path,
    kwarg: str,
    component_cls: type[nn.Module],
    checkpoint_key: str,
) -> None:
    """A bare ``nn.Module`` component is skipped during checkpointing, not an error.

    Both ``pre_processor`` and ``post_processor`` are documented to accept any
    ``nn.Module``, so ``on_save_checkpoint`` must not raise ``AttributeError``
    for a component that doesn't implement the safe-config hooks.
    """
    model = Padim(**{kwarg: component_cls()})
    trainer = Trainer(max_epochs=1, logger=False, barebones=True)
    trainer.strategy.connect(model)
    checkpoint_path = tmp_path / "bare_module.ckpt"

    trainer.save_checkpoint(checkpoint_path)

    checkpoint = AnomalibCheckpointIO().load_checkpoint(checkpoint_path, weights_only=True)
    assert checkpoint_key not in checkpoint


def test_preprocessor_subclass_without_super_init_is_skipped() -> None:
    """A PreProcessor subclass that skips ``super().__init__()`` saves without error.

    Regression test for the documented ``StageSpecificPreProcessor`` pattern
    (see ``docs/.../pre_processor.md``) before it called ``super().__init__()``.
    Such a subclass has no ``.transform`` attribute, so ``checkpoint_config``
    must degrade gracefully via ``getattr(..., None)`` instead of raising.

    Calls ``on_save_checkpoint`` directly rather than going through
    ``Trainer.save_checkpoint``: an ``nn.Module`` submodule that never called
    ``nn.Module.__init__()`` also breaks PyTorch's own ``state_dict()`` walk,
    independently of anything in this checkpoint hook, so routing through the
    full Trainer pipeline would not isolate the behaviour under test.
    """

    class _NoSuperPreProcessor(PreProcessor):
        def __init__(self) -> None:
            self.train_transform = None

    model = Padim(pre_processor=_NoSuperPreProcessor())
    checkpoint: dict = {}

    model.on_save_checkpoint(checkpoint)

    assert checkpoint["anomalib_pre_processor_config"] == {"transform": None}
