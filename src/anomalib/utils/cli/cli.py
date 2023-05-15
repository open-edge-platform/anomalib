"""Anomalib CLI."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import logging
import os
import warnings
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Type, Union
from warnings import warn

from omegaconf.omegaconf import OmegaConf
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.cli import ArgsType, LightningArgumentParser, LightningCLI, SaveConfigCallback

from anomalib.post_processing.normalization import NormalizationMethod
from anomalib.post_processing.post_process import ThresholdMethod
from anomalib.trainer.trainer import AnomalibTrainer
from anomalib.utils.callbacks import ModelCheckpoint, TimerCallback
from anomalib.utils.loggers import configure_logger

logger = logging.getLogger("anomalib.cli")


class AnomalibCLI(LightningCLI):
    """Implementation of a fully configurable CLI tool for anomalib.

    The advantage of this tool is its flexibility to configure the pipeline
    from both the CLI and a configuration file (.yaml or .json). It is even
    possible to use both the CLI and a configuration file simultaneously.
    For more details, the reader could refer to PyTorch Lightning CLI documentation.
    """

    def __init__(
        self,
        model_class: Optional[Union[Type[LightningModule], Callable[..., LightningModule]]] = None,
        datamodule_class: Optional[Union[Type[LightningDataModule], Callable[..., LightningDataModule]]] = None,
        save_config_callback: Optional[Type[SaveConfigCallback]] = SaveConfigCallback,
        save_config_kwargs: Optional[Dict[str, Any]] = None,
        trainer_class: Union[Type[Trainer], Callable[..., Trainer]] = AnomalibTrainer,
        trainer_defaults: Optional[Dict[str, Any]] = None,
        seed_everything_default: Union[bool, int] = True,
        parser_kwargs: Optional[Union[Dict[str, Any], Dict[str, Dict[str, Any]]]] = None,
        subclass_mode_model: bool = False,
        subclass_mode_data: bool = False,
        args: ArgsType = None,
        run: bool = True,
        auto_configure_optimizers: bool = True,
        **kwargs: Any,  # Remove with deprecations of v2.0.0
    ) -> None:
        if trainer_class != AnomalibTrainer:
            warn(f"trainer_class {type(trainer_class)} is not AnomalibTrainer. Setting it to AnomalibTrainer.")
            trainer_class = AnomalibTrainer
        super().__init__(
            model_class,
            datamodule_class,
            save_config_callback,
            save_config_kwargs,
            trainer_class,
            trainer_defaults,
            seed_everything_default,
            parser_kwargs,
            subclass_mode_model,
            subclass_mode_data,
            args,
            run,
            auto_configure_optimizers,
            **kwargs,
        )

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """Add default arguments.

        Args:
            parser (LightningArgumentParser): Lightning Argument Parser.
        """
        group = parser.add_argument_group("Post Processing", description="Normalization and thresholding parameters.")
        group.add_argument(
            "--post_processing.normalization_method", type=NormalizationMethod, default=NormalizationMethod.MIN_MAX
        )
        group.add_argument("--post_processing.threshold_method", type=ThresholdMethod, default=ThresholdMethod.ADAPTIVE)
        group.add_argument("--post_processing.manual_image_threshold", type=Optional[float], default=None)
        group.add_argument("--post_processing.manual_pixel_threshold", type=Optional[float], default=None)
        parser.link_arguments("post_processing.normalization_method", "trainer.normalization_method")
        parser.link_arguments("post_processing.threshold_method", "trainer.threshold_method")
        parser.link_arguments("post_processing.manual_image_threshold", "trainer.manual_image_threshold")
        parser.link_arguments("post_processing.manual_pixel_threshold", "trainer.manual_pixel_threshold")

    def __set_default_root_dir(self) -> None:
        """Sets the default root directory depending on the subcommand type. <train, fit, predict, tune.>."""
        # Get configs.
        subcommand = self.config["subcommand"]
        config = self.config[subcommand]

        # If `resume_from_checkpoint` is not specified, it means that the project has not been created before.
        # Therefore, we need to create the project directory first.
        if config.trainer.resume_from_checkpoint is None:
            root_dir = config.trainer.default_root_dir or "./results"
            model_name = config.model.class_path.split(".")[-1].lower()
            data_name = config.data.class_path.split(".")[-1].lower()
            category = config.data.init_args.category if "category" in config.data.init_args else ""
            time_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            default_root_dir = os.path.join(root_dir, model_name, data_name, category, time_stamp)

        # Otherwise, the assumption is that the project directory has alrady been created.
        else:
            # By default, train subcommand saves the weights to
            #   ./results/<model>/<data>/time_stamp/weights/model.ckpt.
            # For this reason, we set the project directory to the parent directory
            #   that is two-level up.
            default_root_dir = str(Path(config.trainer.resume_from_checkpoint).parent.parent)

        if config.visualization.image_save_path == "":
            self.config[subcommand].visualization.image_save_path = default_root_dir + "/images"
        self.config[subcommand].trainer.default_root_dir = default_root_dir

    def __set_callbacks(self) -> None:
        """Sets the default callbacks used within the pipeline."""
        subcommand = self.config["subcommand"]
        config = self.config[subcommand]

        callbacks = []

        # Model Checkpoint.
        monitor = None
        mode = "max"
        if config.trainer.callbacks is not None:
            # If trainer has callbacks defined from the config file, they have the
            # following format:
            # [{'class_path': 'pytorch_lightning.ca...lyStopping', 'init_args': {...}}]
            callbacks = config.trainer.callbacks

            # Convert to the following format to get `monitor` and `mode` variables
            # {'EarlyStopping': {'monitor': 'pixel_AUROC', 'mode': 'max', ...}}
            callback_args = {c["class_path"].split(".")[-1]: c["init_args"] for c in callbacks}
            if "EarlyStopping" in callback_args:
                monitor = callback_args["EarlyStopping"]["monitor"]
                mode = callback_args["EarlyStopping"]["mode"]

        checkpoint = ModelCheckpoint(
            dirpath=os.path.join(config.trainer.default_root_dir, "weights"),
            filename="model",
            monitor=monitor,
            mode=mode,
            auto_insert_metric_name=False,
        )
        callbacks.append(checkpoint)

        # Add timing to the pipeline.
        callbacks.append(TimerCallback())

        self.config[subcommand].visualization = config.visualization

        # Export to OpenVINO
        if config.export_mode is not None:
            from anomalib.utils.callbacks.export import ExportCallback  # pylint: disable=import-outside-toplevel

            logger.info("Setting model export to %s", config.export_mode)
            callbacks.append(
                ExportCallback(
                    input_size=config.data.init_args.image_size,
                    dirpath=os.path.join(config.trainer.default_root_dir, "compressed"),
                    filename="model",
                    export_mode=config.export_mode,
                )
            )
        else:
            warnings.warn(f"Export option: {config.export_mode} not found. Defaulting to no model export")
        if config.nncf:
            if os.path.isfile(config.nncf) and config.nncf.endswith(".yaml"):
                nncf_module = import_module("anomalib.core.callbacks.nncf_callback")
                nncf_callback = getattr(nncf_module, "NNCFCallback")
                callbacks.append(
                    nncf_callback(
                        config=OmegaConf.load(config.nncf),
                        dirpath=os.path.join(config.trainer.default_root_dir, "compressed"),
                        filename="model",
                    )
                )
            else:
                raise ValueError(f"--nncf expects a path to nncf config which is a yaml file, but got {config.nncf}")

        self.config[subcommand].trainer.callbacks = callbacks

    def before_instantiate_classes(self) -> None:
        """Modify the configuration to properly instantiate classes."""
        self.__set_default_root_dir()
        print("done.")


def main() -> None:
    """Trainer via Anomalib CLI."""
    configure_logger()
    AnomalibCLI()


if __name__ == "__main__":
    main()
