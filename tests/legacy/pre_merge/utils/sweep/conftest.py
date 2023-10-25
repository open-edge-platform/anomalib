"""Fixtures for the sweep tests."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from tempfile import TemporaryDirectory
from typing import Optional, Union

import pytest
from lightning.pytorch import Trainer
from omegaconf import DictConfig, ListConfig

from anomalib.config import get_configurable_parameters
from anomalib.data import get_datamodule
from anomalib.engine import Engine
from anomalib.models import get_model
from anomalib.utils.callbacks import get_callbacks


def get_model_config(
    project_path: str,
    model_name: str,
    dataset_path: str,
    category: str,
    task: str = "classification",
    export_mode: Optional[str] = None,
):
    model_config = get_configurable_parameters(model_name=model_name)
    model_config.trainer.default_root_dir = project_path
    model_config.data.init_args.task = task
    model_config.data.init_args.root = dataset_path
    model_config.data.init_args.category = category
    model_config.trainer.fast_dev_run = True
    model_config.trainer.max_epochs = 1
    model_config.trainer.devices = 1
    model_config.trainer.accelerator = "gpu"
    model_config.optimization.export_mode = export_mode
    return model_config


@pytest.fixture(scope="package")
def generate_results_dir():
    with TemporaryDirectory() as project_path:

        def make(
            model_name: str,
            dataset_path: str,
            category: str,
            task: str = "classification",
            export_mode: Optional[str] = None,
        ) -> Union[DictConfig, ListConfig]:
            # then train the model
            model_config = get_model_config(
                project_path=project_path,
                model_name=model_name,
                dataset_path=dataset_path,
                category=category,
                task=task,
                export_mode=export_mode,
            )
            model = get_model(model_config)
            datamodule = get_datamodule(model_config)
            callbacks = get_callbacks(model_config)
            engine = Engine(**model_config.trainer, logger=False, callbacks=callbacks)
            engine.fit(model=model, datamodule=datamodule)

            return model_config

        yield make
