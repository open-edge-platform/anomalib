from argparse import ArgumentParser
import os

import torch
from pytorch_lightning import Trainer

from anomalib.config.config import get_configurable_parameters
from anomalib.datasets import get_datamodule
from anomalib.models import get_model
import time


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="stfpm", help="Name of the algorithm to train/test")
    parser.add_argument("--model_config_path", type=str, required=False, help="Path to a model config file")
    parser.add_argument("--weight_file", type=str, default="./results/weights/model.ckpt")
    args = parser.parse_args()
    return args


args = get_args()
config = get_configurable_parameters(model_name=args.model, model_config_path=args.model_config_path)
datamodule = get_datamodule(config.dataset)
datamodule.setup()
train_loader = datamodule.train_dataloader()
model = get_model(config.model, train_loader)

trainer = Trainer(callbacks=model.callbacks, **config.trainer)
trainer.test(model=model, datamodule=datamodule)
