from unittest import mock

import numpy as np
import pytest
from omegaconf import OmegaConf

from anomalib.core.callbacks.visualizer_callback import VisualizerCallback
from anomalib.hpo.sweep import run_sweep
from anomalib.hpo.sweep.config import (
    IncorrectHPOConfiguration,
    MissingHPOConfiguration,
    get_experiment,
    validate_config,
    validate_search_params,
)
from tests.hpo.sweep.dummy_lightning_model import DummyModel, XORDataModule


@mock.patch("anomalib.loggers.sigopt.sigopt")
def test_validate_params(sigopt):
    """This tests whether validation of hpo params works"""
    connection = sigopt.Connection()
    config = OmegaConf.load("tests/hpo/sweep/test_config.yaml")

    # check when correct config is passed
    get_experiment(connection=connection, config=config)

    # check when no HPO is passed to in config
    config.pop("hyperparameter_search", None)
    with pytest.raises(MissingHPOConfiguration) as info:
        validate_config(config)
    assert "does not contain parameters for hyperparameter optimization." in str(info.value)

    config["hyperparameter_search"] = {}
    # empty metric
    config.hyperparameter_search.metric = {}
    with pytest.raises(IncorrectHPOConfiguration) as info:
        validate_config(config)
    assert "Optimization metric should use one metric" in str(info.value)

    # missing keys
    config.hyperparameter_search.metric = {"name": "test_metric", "goal": "minimize"}
    with pytest.raises(KeyError) as info:
        validate_config(config)
    assert "Missing key objective" in str(info.value)
    config.hyperparameter_search.metric = {"val": "test_metric", "objective": "minimize"}
    with pytest.raises(KeyError) as info:
        validate_config(config)
    assert "Missing key name" in str(info.value)

    # incorrect objective
    config.hyperparameter_search.metric = {"name": "test_objective", "objective": "max"}
    with pytest.raises(IncorrectHPOConfiguration) as info:
        validate_config(config)
    assert "Objective should be one of [maximize, minimize]" in str(info.value)

    # multiple keys
    config.hyperparameter_search.metric = {
        "name": "test_metric1",
        "objective": "minimize",
        "name2": "test_metric2",
        "objective2": "minimize",
    }
    with pytest.raises(IncorrectHPOConfiguration) as info:
        validate_config(config)
    assert "Optimization metric should use one metric." in str(info.value)


def test_validate_dtypes():
    config = OmegaConf.create({"lr": {"type": "double", "min": 0, "max": 1}})
    with pytest.raises(IncorrectHPOConfiguration) as info:
        for param in config.values():
            validate_search_params(params=param)
        assert "Type mismatch in parameter configuration. Expected float" in str(info.value)

    config = OmegaConf.create({"patience": {"type": "int", "min": 0.0, "max": 1.0}})
    with pytest.raises(IncorrectHPOConfiguration) as info:
        for param in config.values():
            validate_search_params(params=param)
        assert "Type mismatch in parameter configuration. Expected integer" in str(info.value)


def mock_get_model(config):
    if config.model.backbone == "XOR":
        raise Exception("HPO is not suggesting backbones")
    return DummyModel(hparams=config)


def mock_get_datamodule(config):
    return XORDataModule()


def mock_connection():
    class Name:
        """Probably a bit complicated way to do this but this is a way around having a `name` parameter in mock which
        does not collide with default parameter."""

        def __init__(self):
            self.name = "loss"

    budget = 3
    experiment_mocks = [
        mock.Mock(
            progress=mock.Mock(
                observation_count=i,
            ),
            observation_budget=budget,
            metric=Name(),
            id="dummy_id",
        )
        for i in range(budget + 1)
    ]
    connection = mock.Mock()
    experiment_resource = mock.Mock(
        create=mock.Mock(side_effect=experiment_mocks),
        fetch=mock.Mock(side_effect=experiment_mocks),
        suggestions=mock.Mock(
            return_value=mock.Mock(
                create=mock.Mock(
                    side_effect=[
                        mock.Mock(
                            assignments={
                                "model.lr": np.random.uniform(low=1e-3, high=1.0),
                                "model.momentum": np.random.uniform(low=0, high=1.0),
                                "model.patience": np.random.randint(low=1, high=10),
                                "model.weight_decay": np.random.uniform(low=1e-5, high=1e-3),
                                "model.backbone": model,
                            }
                        )
                        for model in ["DummyModel1", "DummyModel2"] * 2
                    ]
                )
            )
        ),
    )
    connection.experiments = mock.Mock(return_value=experiment_resource)
    return connection


def mock_test_return_empty(*args, **kwargs):
    return [{}]


def mock_test_return_incorrect(*args, **kwargs):
    return [{"accuracy": 0.0}]


@mock.patch("anomalib.hpo.sweep.sweep.Connection", mock_connection)
@mock.patch("anomalib.hpo.sweep.sweep.get_datamodule", mock_get_datamodule)
@mock.patch("anomalib.hpo.sweep.sweep.get_model", mock_get_model)
def test_hpo():
    """This tests the sweep function"""

    config = OmegaConf.load("tests/hpo/sweep/test_config.yaml")

    run_sweep(config=config)

    # test that value error is thrown
    with pytest.raises(ValueError):
        with mock.patch("anomalib.hpo.sweep.sweep.Trainer.test", mock_test_return_empty):
            run_sweep(config=config)

    # If wrong key is returned
    with pytest.raises(KeyError):
        with mock.patch("anomalib.hpo.sweep.sweep.Trainer.test", mock_test_return_incorrect):
            run_sweep(config=config)


def check_visualizer_callback_exists(callbacks, **kwargs):
    # raise a key error if visualizer is not removed and exit the test
    # trainer is set to None if no key error is raised.
    # This will cause the sweep to fail with AttributeError thus ending the test before entire sweep is run
    for index, callback in enumerate(callbacks):
        if isinstance(callback, VisualizerCallback):
            raise KeyError("Visualizer Callback exists in the model")
    return None


@mock.patch("anomalib.hpo.sweep.sweep.Connection", mock_connection)
@mock.patch("anomalib.hpo.sweep.sweep.get_datamodule", mock_get_datamodule)
@mock.patch("anomalib.hpo.sweep.sweep.get_model", mock_get_model)
def test_remove_visualizer_callback():
    """Tests whether visualizer callback is removed during hpo"""
    config = OmegaConf.load("tests/hpo/sweep/test_config.yaml")
    # raise a key error if visualizer is not removed and exit the test.
    # If it is removed then trainer object is set to none and the sweep function returns AttributeError
    with pytest.raises(AttributeError):
        with mock.patch("anomalib.hpo.sweep.sweep.Trainer", check_visualizer_callback_exists):
            run_sweep(config)
