"""
Anomalib Traning Script.
    This script reads the name of the model or config file
    from command line, train/test the anomaly model to get
    quantitative and qualitative results.
"""
from argparse import ArgumentParser, Namespace
import os

from pytorch_lightning import Trainer, seed_everything
from torch import load

from anomalib.config.config import get_configurable_parameters
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
    parser.add_argument("--model", type=str, default="stfpm", help="Name of the algorithm to train/test")
    parser.add_argument("--model_config_path", type=str, required=False, help="Path to a model config file")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    config = get_configurable_parameters(model_name=args.model, model_config_path=args.model_config_path)

    if config.project.seed != 0:
        seed_everything(config.project.seed)

    datamodule = get_datamodule(config)
    model = get_model(config)
    logger = get_logger(config)

<<<<<<< HEAD
    if "init_weights" in config.keys():
        model.load_state_dict(load(os.path.join(config.project.path, config.init_weights))['state_dict'], strict=False)

    trainer = Trainer(**config.trainer, logger=logger)
=======
    trainer = Trainer(**config.trainer, logger=logger, callbacks=model.callbacks)
>>>>>>> development
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)
