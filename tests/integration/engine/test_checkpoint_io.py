# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for weights-only Engine checkpoint loading."""

from collections.abc import Callable
from pathlib import Path

from torchvision.transforms.v2 import Resize

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.engine.plugins import AnomalibCheckpointIO
from anomalib.models import Padim, Patchcore
from anomalib.pre_processing import PreProcessor


class TestCheckpointIO:
    """Test Engine checkpoint restore under the shared-fixture workflow."""

    @staticmethod
    def test_engine_test_loads_shared_checkpoint_weights_only(
        ckpt_path: Callable[[str], Path],
        project_path: Path,
        mvtecad_path: Path,
    ) -> None:
        """Engine ``ckpt_path`` restores via AnomalibCheckpointIO without a second fit.

        Uses the session-scoped ``ckpt_path`` resolver from ``tests.fixtures.models``
        so this does not retrain Padim when other integration modules already have.
        """
        checkpoint_path = ckpt_path("Padim")
        model = Padim()
        engine = Engine(
            default_root_dir=project_path,
            fast_dev_run=True,
            devices=1,
            logger=False,
        )
        datamodule = MVTecAD(root=mvtecad_path, category="dummy")

        results = engine.test(model=model, datamodule=datamodule, ckpt_path=str(checkpoint_path))

        assert results
        assert isinstance(engine.trainer.strategy.checkpoint_io, AnomalibCheckpointIO)

    @staticmethod
    def test_load_from_checkpoint_reads_shared_checkpoint(
        ckpt_path: Callable[[str], Path],
    ) -> None:
        """``load_from_checkpoint`` restores a trained Padim under weights_only=True."""
        checkpoint_path = ckpt_path("Padim")

        loaded = Padim.load_from_checkpoint(checkpoint_path)

        assert isinstance(loaded, Padim)

    @staticmethod
    def test_predict_restores_patchcore_with_custom_pre_processor(
        project_path: Path,
        mvtecad_path: Path,
    ) -> None:
        """A PatchCore checkpoint trained with a custom PreProcessor restores for prediction.

        Regression test for https://github.com/open-edge-platform/anomalib/issues/3612,
        where restoring such a checkpoint through ``Engine.predict(ckpt_path=...)``
        failed with a ``weights_only`` unpickling error.
        """
        image_size = (64, 96)

        def build_model() -> Patchcore:
            return Patchcore(
                backbone="resnet18",
                pre_trained=False,
                coreset_sampling_ratio=0.1,
                pre_processor=PreProcessor(transform=Resize(size=image_size)),
            )

        root_dir = project_path / "custom_pre_processor"
        datamodule = MVTecAD(root=mvtecad_path, category="dummy", train_batch_size=4, eval_batch_size=4)
        train_engine = Engine(default_root_dir=root_dir, logger=False, devices=1)
        train_engine.fit(model=build_model(), datamodule=datamodule)
        checkpoint_path = train_engine.best_model_path
        assert checkpoint_path

        predict_engine = Engine(default_root_dir=root_dir, logger=False, devices=1)
        predictions = predict_engine.predict(
            model=build_model(),
            ckpt_path=checkpoint_path,
            data_path=mvtecad_path / "dummy" / "test",
        )

        assert predictions
        assert isinstance(predict_engine.trainer.strategy.checkpoint_io, AnomalibCheckpointIO)
        assert predictions[0].anomaly_map.shape[-2:] == image_size
