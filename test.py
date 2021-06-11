from anomalib.helpers.sigopt_logger import SigoptLogger
from argparse import ArgumentParser

import torch
from pytorch_lightning import Trainer

from anomalib.config import get_configurable_parameters
from anomalib.datasets import get_datamodule
from anomalib.models import get_model


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="stfpm", help="Name of the algorithm to train/test")
    parser.add_argument("--model_config_path", type=str, required=False, help="Path to a model config file")
    parser.add_argument("--weight_file", type=str, default="./results/weights/model.ckpt")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    config = get_configurable_parameters(model_name=args.model, model_config_path=args.model_config_path)
    datamodule = get_datamodule(config.dataset)

    model = get_model(config.model)
    # TODO: load_from_checkpoint doesn't properly load the weights!!!
    # model.load_from_checkpoint(checkpoint_path="./results/weights/pl_model.pth")
    model.load_state_dict(torch.load(args.weight_file)["state_dict"])

    logger = SigoptLogger(project="anomaly", name=f"{args.model}_test")

    trainer = Trainer(callbacks=model.callbacks, **config.trainer, logger=logger)
    trainer.test(model=model, datamodule=datamodule)
