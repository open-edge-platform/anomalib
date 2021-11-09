"""
Anomalib Traning Script.
This script reads the name of the model or config file
from command line, train/test the anomaly model to get
quantitative and qualitative results.
"""
from argparse import ArgumentParser, Namespace

from pytorch_lightning import Trainer, seed_everything

from anomalib.config import get_configurable_parameters
from anomalib.core.callbacks import get_callbacks
from anomalib.datasets import get_datamodule
from anomalib.loggers import get_logger
from anomalib.models import get_model


def get_args() -> Namespace:
    """
    Get command line arguments.

    Returns:
        Namespace: List of arguments.
    """
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="padim", help="Name of the algorithm to train/test")
    parser.add_argument("--model_config_path", type=str, required=False, help="Path to a model config file")

    return parser.parse_args()


def train():
    """
    Train an anomaly classification or segmentation model based on a provided configuration file.
    """
    args = get_args()
    config = get_configurable_parameters(model_name=args.model, model_config_path=args.model_config_path)

    if config.project.seed != 0:
        seed_everything(config.project.seed)

    datamodule = get_datamodule(config)
    model = get_model(config)
    logger = get_logger(config)

    callbacks = get_callbacks(config)

    trainer = Trainer(**config.trainer, logger=logger, callbacks=callbacks)
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    train()
