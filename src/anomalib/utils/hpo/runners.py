"""Sweep Backends."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import gc
from warnings import warn

import torch
from lightning_utilities.core.imports import module_available
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning.loggers import CometLogger, WandbLogger

from anomalib.config import update_input_size_config
from anomalib.data import get_datamodule
from anomalib.models import get_model
from anomalib.trainer import AnomalibTrainer
from anomalib.utils.sweep import flatten_sweep_params, set_in_nested_config

from .config import flatten_hpo_params

try:
    import wandb
except ImportError:
    warn("wandb not installed. Please install to use these features.")


try:
    from comet_ml import Optimizer
except ImportError:
    warn("comet_ml not installed. Please install to use these features.")


class WandbSweep:
    """wandb sweep.

    Args:
        config (DictConfig): Original model configuration.
        sweep_config (DictConfig): Sweep configuration.
        entity (str, optional): Username or workspace to send the project to. Defaults to None.
    """

    def __init__(
        self,
        config: DictConfig | ListConfig,
        sweep_config: DictConfig | ListConfig,
        entity: str | None = None,
    ) -> None:
        if not module_available("wandb"):
            raise ImportError("wandb not installed. Please install to use these features.")
        self.config = config
        self.sweep_config = sweep_config
        self.observation_budget = sweep_config.observation_budget
        self.entity = entity
        if "observation_budget" in self.sweep_config.keys():
            # this instance check is to silence mypy.
            if isinstance(self.sweep_config, DictConfig):
                self.sweep_config.pop("observation_budget")

    def run(self) -> None:
        """Run the sweep."""
        flattened_hpo_params = flatten_hpo_params(self.sweep_config.parameters)
        self.sweep_config.parameters = flattened_hpo_params
        sweep_id = wandb.sweep(
            OmegaConf.to_object(self.sweep_config),
            project=f"{self.config.model.name}_{self.config.dataset.name}",
            entity=self.entity,
        )
        wandb.agent(sweep_id, function=self.sweep, count=self.observation_budget)

    def sweep(self) -> None:
        """Method to load the model, update config and call fit. The metrics are logged to ```wandb``` dashboard."""
        wandb_logger = WandbLogger(config=flatten_sweep_params(self.sweep_config), log_model=False)
        sweep_config = wandb_logger.experiment.config

        for param in sweep_config.keys():
            set_in_nested_config(self.config, param.split("."), sweep_config[param])
        config = update_input_size_config(self.config)

        model = get_model(config)
        datamodule = get_datamodule(config)

        # Disable saving checkpoints as all checkpoints from the sweep will get uploaded
        config.trainer.enable_checkpointing = False

        trainer = AnomalibTrainer(
            **config.trainer,
            **config.post_processing,
            logger=wandb_logger,
            task_type=config.dataset.task,
            image_metrics=config.metrics.get("image", None),
            pixel_metrics=config.metrics.get("pixel", None),
        )
        trainer.fit(model, datamodule=datamodule)

        del model
        gc.collect()
        torch.cuda.empty_cache()


class CometSweep:
    """comet sweep.

    Args:
        config (DictConfig): Original model configuration.
        sweep_config (DictConfig): Sweep configuration.
        entity (str, optional): Username or workspace to send the project to. Defaults to None.
    """

    def __init__(
        self,
        config: DictConfig | ListConfig,
        sweep_config: DictConfig | ListConfig,
        entity: str | None = None,
    ) -> None:
        if not module_available("comet_ml"):
            raise ImportError("comet_ml not installed. Please install to use these features.")
        self.config = config
        self.sweep_config = sweep_config
        self.entity = entity

    def run(self) -> None:
        """Run the sweep."""
        flattened_hpo_params = flatten_hpo_params(self.sweep_config.parameters)
        self.sweep_config.parameters = flattened_hpo_params

        # comet's Optimizer takes dict as an input, not DictConfig
        std_dict = OmegaConf.to_object(self.sweep_config)

        opt = Optimizer(std_dict)

        project_name = f"{self.config.model.name}_{self.config.dataset.name}"

        for experiment in opt.get_experiments(project_name=project_name):
            comet_logger = CometLogger(workspace=self.entity)

            # allow pytorch-lightning to use the experiment from optimizer
            comet_logger._experiment = experiment  # pylint: disable=protected-access
            run_params = experiment.params
            for param in run_params.keys():
                # this check is needed as comet also returns model and sweep_config as keys
                if param in self.sweep_config.parameters.keys():
                    set_in_nested_config(self.config, param.split("."), run_params[param])
            config = update_input_size_config(self.config)

            model = get_model(config)
            datamodule = get_datamodule(config)

            # Disable saving checkpoints as all checkpoints from the sweep will get uploaded
            config.trainer.enable_checkpointing = False

            trainer = AnomalibTrainer(
                **config.trainer,
                **config.post_processing,
                logger=comet_logger,
                task_type=config.dataset.task,
                image_metrics=config.metrics.get("image", None),
                pixel_metrics=config.metrics.get("pixel", None),
            )
            trainer.fit(model, datamodule=datamodule)
