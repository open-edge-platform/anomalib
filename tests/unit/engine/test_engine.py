# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Test Engine Module."""

from pathlib import Path

import pytest
import yaml
from lightning import seed_everything

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Padim

# Tolerance threshold for comparing metric values between normal and barebones modes


class TestEngine:
    """Test Engine."""

    @pytest.fixture()
    @staticmethod
    def fxt_full_config_path(tmp_path: Path) -> Path:
        """Fixture full configuration examples."""
        config_str = """
        seed_everything: true
        trainer:
            accelerator: auto
            strategy: auto
            devices: auto
            num_nodes: 1
            precision: null
            logger: null
            callbacks: null
            fast_dev_run: false
            max_epochs: null
            min_epochs: null
            max_steps: -1
            min_steps: null
            max_time: null
            limit_train_batches: null
            limit_val_batches: null
            limit_test_batches: null
            limit_predict_batches: null
            overfit_batches: 0.0
            val_check_interval: null
            check_val_every_n_epoch: 1
            num_sanity_val_steps: null
            log_every_n_steps: null
            enable_checkpointing: null
            enable_progress_bar: null
            enable_model_summary: null
            accumulate_grad_batches: 1
            gradient_clip_val: null
            gradient_clip_algorithm: null
            deterministic: null
            benchmark: null
            inference_mode: true
            use_distributed_sampler: true
            profiler: null
            detect_anomaly: false
            barebones: false
            plugins: null
            sync_batchnorm: false
            reload_dataloaders_every_n_epochs: 0
        logging:
            log_graph: false
        default_root_dir: results
        ckpt_path: null
        model:
            class_path: anomalib.models.Padim
            init_args:
                backbone: resnet18
                layers:
                - layer1
                - layer2
                - layer3
                pre_trained: true
                n_features: null
        data:
            class_path: anomalib.data.MVTecAD
            init_args:
                root: datasets/MVTecAD
                category: bottle
                train_batch_size: 32
                eval_batch_size: 32
                num_workers: 8
                test_split_mode: FROM_DIR
                test_split_ratio: 0.2
                val_split_mode: SAME_AS_TEST
                val_split_ratio: 0.5
                seed: null
        """
        config_dict = yaml.safe_load(config_str)
        config_file = tmp_path / "config.yaml"
        with config_file.open(mode="w") as file:
            yaml.dump(config_dict, file)
        return config_file

    @staticmethod
    def test_from_config(fxt_full_config_path: Path) -> None:
        """Test Engine.from_config."""
        with pytest.raises(FileNotFoundError):
            Engine.from_config(config_path="wrong_configs.yaml")

        engine, model, datamodule = Engine.from_config(config_path=fxt_full_config_path)
        assert engine is not None
        assert isinstance(engine, Engine)
        assert model is not None
        assert isinstance(model, Padim)
        assert datamodule is not None
        assert isinstance(datamodule, MVTecAD)
        assert datamodule.train_batch_size == 32
        assert datamodule.num_workers == 8

        # Override batch_size & num_workers
        override_kwargs = {
            "data.train_batch_size": 1,
            "data.num_workers": 1,
        }
        engine, model, datamodule = Engine.from_config(config_path=fxt_full_config_path, **override_kwargs)
        assert datamodule.train_batch_size == 1
        assert datamodule.num_workers == 1

    @staticmethod
    def test_barebones_mode_metrics_and_checkpointing(tmp_path: Path, dataset_path: Path) -> None:
        """Test that barebones mode returns the same metrics and disables checkpointing.

        This test verifies that:
        1. Barebones mode and normal mode return the same metric values
        2. Both modes return the same set of metric keys
        3. Metrics are properly captured in barebones mode despite logging being disabled
        4. Normal mode (barebones=False) creates checkpoint files
        5. Barebones mode (barebones=True) does not create checkpoint files
        """
        datamodule = MVTecAD(root=dataset_path / "mvtecad", category="dummy")

        # Test with normal mode
        seed_everything(42, workers=True)
        model_normal = Padim()
        engine_normal = Engine(default_root_dir=tmp_path / "normal")
        engine_normal.fit(model=model_normal, datamodule=datamodule)
        results_normal = engine_normal.test(model=model_normal, datamodule=datamodule)

        # Test with barebones mode
        seed_everything(42, workers=True)
        model_barebones = Padim()
        engine_barebones = Engine(default_root_dir=tmp_path / "barebones", barebones=True)
        engine_barebones.fit(model=model_barebones, datamodule=datamodule)
        results_barebones = engine_barebones.test(model=model_barebones, datamodule=datamodule)

        # Verify both modes return results
        assert results_normal
        assert results_barebones
        assert len(results_normal) > 0
        assert len(results_barebones) > 0

        # Extract metrics
        metrics_normal = results_normal[0]
        metrics_barebones = results_barebones[0]

        # Verify both have the same metric keys
        assert set(metrics_normal.keys()) == set(metrics_barebones.keys())

        # Verify expected metrics are present
        expected_metrics = {"image_AUROC", "image_F1Score", "pixel_AUROC", "pixel_F1Score"}
        assert expected_metrics.issubset(set(metrics_normal.keys()))

        # Verify metric values are the same
        for metric_name in metrics_normal:
            value_normal = metrics_normal[metric_name]
            value_barebones = metrics_barebones[metric_name]

            if hasattr(value_normal, "item"):
                value_normal = value_normal.item()
            if hasattr(value_barebones, "item"):
                value_barebones = value_barebones.item()

            assert abs(value_normal - value_barebones) < 0.01

        # Verify checkpoint behavior
        normal_checkpoints = list((tmp_path / "normal").rglob("*.ckpt"))
        barebones_checkpoints = list((tmp_path / "barebones").rglob("*.ckpt"))

        # Verify normal mode (barebones=False) creates checkpoints
        assert len(normal_checkpoints) > 0, "Normal mode (barebones=False) should create checkpoint files"

        # Verify barebones mode (barebones=True) does not create checkpoints
        assert len(barebones_checkpoints) == 0, "Barebones mode (barebones=True) should not create checkpoint files"
