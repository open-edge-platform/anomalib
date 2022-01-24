"""Common helpers for both nightly and pre-merge model tests."""

# Copyright (C) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from typing import Dict, Tuple, Union

import numpy as np
from omegaconf import DictConfig, ListConfig
from pytorch_lightning import LightningDataModule, Trainer

from anomalib.config import get_configurable_parameters, update_nncf_config
from anomalib.core.callbacks import get_callbacks
from anomalib.core.callbacks.visualizer_callback import VisualizerCallback
from anomalib.core.model.anomaly_module import AnomalyModule
from anomalib.data import get_datamodule
from anomalib.models import get_model


def setup(
    model_name: str,
    dataset_path: str,
    project_path: str,
    nncf: bool,
    category: str,
    score_type: str = None,
    weight_file: str = "weights/model.ckpt",
    fast_run: bool = False,
) -> Tuple[AnomalyModule, Union[DictConfig, ListConfig], LightningDataModule, Trainer]:
    """Train the model based on the parameters passed.

    Args:
        model_name (str): Name of the model to train.
        dataset_path (str): Location of the dataset.
        project_path (str): Path to temporary project folder.
        nncf (bool): Add nncf callback.
        category (str): Category to train on.
        score_type (str, optional): Only used for DFM. Defaults to None.
        weight_file (str, optional): Path to weight file.
        fast_run (bool, optional): If set to true, the model trains for only 1 epoch. We train for one epoch as
            this ensures that both anomalous and non-anomalous images are present in the validation step.

    Returns:
        Tuple[AnomalyModule, DictConfig, LightningDataModule, Trainer]: trained model, updated config, datamodule, trainer object
    """
    config = get_configurable_parameters(model_name=model_name)
    if score_type is not None:
        config.model.score_type = score_type
    config.project.seed = 1234
    config.dataset.category = category
    config.dataset.path = dataset_path

    # If weight file is empty, remove the key from config
    if "weight_file" in config.model.keys() and weight_file == "":
        config.model.pop("weight_file")
    else:
        config.model.weight_file = weight_file

    if nncf:
        config.optimization.nncf.apply = True
        config = update_nncf_config(config)
        config.init_weights = None

    # reassign project path as config is updated in `update_config_for_nncf`
    config.project.path = project_path

    datamodule = get_datamodule(config)
    model = get_model(config)

    callbacks = get_callbacks(config)

    # Force saving the weights after 1 epoch of training. This is used for testing model loading on pre-merge.
    if "early_stopping" in config.model.keys() and fast_run == True:
        config.model.early_stopping.metric = None
    callbacks = get_callbacks(config)

    for index, callback in enumerate(callbacks):
        if isinstance(callback, VisualizerCallback):
            callbacks.pop(index)
            break

    # Train the model.
    if fast_run:
        config.trainer.max_epochs = 1

    trainer = Trainer(callbacks=callbacks, **config.trainer)
    trainer.fit(model=model, datamodule=datamodule)
    return model, config, datamodule, trainer


def model_load_test(config: Union[DictConfig, ListConfig], datamodule: LightningDataModule, results: Dict):
    """Create a new model based on the weights specified in config.

    Args:
        config ([Union[DictConfig, ListConfig]): Model config.
        datamodule (LightningDataModule): Dataloader
        results (Dict): Results from original model.

    """
    loaded_model = get_model(config)  # get new model

    callbacks = get_callbacks(config)

    for index, callback in enumerate(callbacks):
        # Remove visualizer callback as saving results takes time
        if isinstance(callback, VisualizerCallback):
            callbacks.pop(index)
            break

    # create new trainer object with LoadModel callback (assumes it is present)
    trainer = Trainer(callbacks=callbacks, **config.trainer)
    # Assumes the new model has LoadModel callback and the old one had ModelCheckpoint callback
    new_results = trainer.test(model=loaded_model, datamodule=datamodule)[0]
    assert np.isclose(
        results["image_AUROC"], new_results["image_AUROC"]
    ), "Loaded model does not yield close performance results"
    if config.dataset.task == "segmentation":
        assert np.isclose(
            results["pixel_AUROC"], new_results["pixel_AUROC"]
        ), "Loaded model does not yield close performance results"
