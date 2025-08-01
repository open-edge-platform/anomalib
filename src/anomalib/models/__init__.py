# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Anomaly detection models.

This module contains all the anomaly detection models available in anomalib.

Example:
    >>> from anomalib.data import MVTecAD
    >>> from anomalib.models import Padim
    >>> from anomalib.engine import Engine

    >>> # Initialize model and datamodule
    >>> datamodule = MVTecAD()
    >>> model = Padim()

    >>> # Train using the engine
    >>> engine = Engine()
    >>> engine.fit(model=model, datamodule=datamodule)

The module provides both image and video anomaly detection models:

Image Models:
    - CFA (:class:`anomalib.models.image.Cfa`)
    - Cflow (:class:`anomalib.models.image.Cflow`)
    - CSFlow (:class:`anomalib.models.image.Csflow`)
    - DFKDE (:class:`anomalib.models.image.Dfkde`)
    - DFM (:class:`anomalib.models.image.Dfm`)
    - DRAEM (:class:`anomalib.models.image.Draem`)
    - DSR (:class:`anomalib.models.image.Dsr`)
    - EfficientAd (:class:`anomalib.models.image.EfficientAd`)
    - FastFlow (:class:`anomalib.models.image.Fastflow`)
    - FRE (:class:`anomalib.models.image.Fre`)
    - GANomaly (:class:`anomalib.models.image.Ganomaly`)
    - PaDiM (:class:`anomalib.models.image.Padim`)
    - PatchCore (:class:`anomalib.models.image.Patchcore`)
    - Reverse Distillation (:class:`anomalib.models.image.ReverseDistillation`)
    - STFPM (:class:`anomalib.models.image.Stfpm`)
    - SuperSimpleNet (:class:`anomalib.models.image.Supersimplenet`)
    - UFlow (:class:`anomalib.models.image.Uflow`)
    - VLM-AD (:class:`anomalib.models.image.VlmAd`)
    - WinCLIP (:class:`anomalib.models.image.WinClip`)

Video Models:
    - AI-VAD (:class:`anomalib.models.video.AiVad`)
"""

import logging
from importlib import import_module

from jsonargparse import Namespace
from omegaconf import DictConfig, OmegaConf

from anomalib.models.components import AnomalibModule
from anomalib.utils.path import convert_snake_to_pascal_case, convert_to_snake_case, convert_to_title_case

from .image import (
    Cfa,
    Cflow,
    Csflow,
    Dfkde,
    Dfm,
    Dinomaly,
    Draem,
    Dsr,
    EfficientAd,
    Fastflow,
    Fre,
    Ganomaly,
    Padim,
    Patchcore,
    ReverseDistillation,
    Stfpm,
    Supersimplenet,
    Uflow,
    UniNet,
    VlmAd,
    WinClip,
)
from .video import AiVad, Fuvas

# Whitelist of allowed modules for dynamic imports
ALLOWED_MODULES = {
    "anomalib.models",
    "anomalib.models.image",
    "anomalib.models.video",
    "anomalib.models.components",
}


class UnknownModelError(ModuleNotFoundError):
    pass


__all__ = [
    "Cfa",
    "Cflow",
    "Csflow",
    "Dfkde",
    "Dfm",
    "Draem",
    "Dinomaly",
    "Dsr",
    "EfficientAd",
    "Fastflow",
    "Fre",
    "Ganomaly",
    "Padim",
    "Patchcore",
    "ReverseDistillation",
    "Stfpm",
    "Supersimplenet",
    "Uflow",
    "UniNet",
    "VlmAd",
    "WinClip",
    "AiVad",
    "Fuvas",
]

logger = logging.getLogger(__name__)


def list_models(case: str = "snake") -> set[str]:
    """List available anomaly detection models.

    Returns a set of model names in the specified format that are available in the
    anomalib library. This includes both image and video anomaly detection models.

    Args:
        case (str): The format to return model names in. Options are:
            - "snake_case": Returns names in snake_case format (e.g. "efficient_ad")
            - "original": Returns the original PascalCase class names (e.g. "EfficientAd")
            Defaults to "snake_case".

    Returns:
        set[str]: Set of available model names in the specified format.

    Example:
        Get all available models in different formats:

        >>> from anomalib.models import list_models
        >>> # Get models in snake_case format
        >>> models = list_models(case="snake")
        >>> print(sorted(list(models)))  # doctest: +NORMALIZE_WHITESPACE
        ['ai_vad', 'cfa', 'cflow', 'csflow', 'dfkde', 'dfm', 'draem',
         'efficient_ad', 'fastflow', 'fre', 'ganomaly', 'padim', 'patchcore',
         'reverse_distillation', 'stfpm', 'uflow', 'vlm_ad', 'winclip']

        >>> # Get models in original PascalCase format
        >>> models = list_models(case="pascal")
        >>> print(sorted(list(models)))  # doctest: +NORMALIZE_WHITESPACE
        ['AiVad', 'Cfa', 'Cflow', 'Csflow', 'Dfkde', 'Dfm', 'Draem',
         'EfficientAd', 'Fastflow', 'Fre', 'Ganomaly', 'Padim', 'Patchcore',
         'ReverseDistillation', 'Stfpm', 'Uflow', 'VlmAd', 'WinClip']

        >>> # Get models in title case format
        >>> models = list_models(case="title")
        >>> print(sorted(list(models)))  # doctest: +NORMALIZE_WHITESPACE
        ['Ai Vad', 'Cfa', 'Cflow', 'Csflow', 'Dfkde', 'Dfm', 'Draem',
         'Efficient Ad', 'Fastflow', 'Fre', 'Ganomaly', 'Padim', 'Patchcore',
         'Reverse Distillation', 'Stfpm', 'Uflow', 'Vlm Ad', 'Win Clip']

    Note:
        The returned model names can be used with :func:`get_model` to instantiate
        the corresponding model class.
    """
    if case not in {"snake", "pascal", "title"}:
        msg = f"Unsupported format: {case}. Must be one of: snake, pascal, title"
        raise ValueError(msg)

    models = {cls.__name__ for cls in AnomalibModule.__subclasses__() if cls.__name__ != "AnomalyModule"}

    if case == "snake":
        return {convert_to_snake_case(name) for name in models}

    if case == "title":
        return {convert_to_title_case(name) for name in models}

    return models


def _get_model_class_by_name(name: str) -> type[AnomalibModule]:
    """Retrieve an anomaly model class based on its name.

    This internal function takes a model name and returns the corresponding model class.
    The name matching is case-insensitive and supports both snake_case and PascalCase
    formats.

    Args:
        name (str): Name of the model to retrieve. Can be in snake_case (e.g.
            ``"efficient_ad"``) or PascalCase (e.g. ``"EfficientAd"``). The name is
            case-insensitive.

    Raises:
        UnknownModelError: If no model is found matching the provided name. The error
            message includes the list of available models.

    Returns:
        type[AnomalibModule]: Model class that inherits from ``AnomalibModule``.

    Examples:
        >>> from anomalib.models import _get_model_class_by_name
        >>> model_class = _get_model_class_by_name("padim")
        >>> model_class.__name__
        'Padim'
        >>> model_class = _get_model_class_by_name("efficient_ad")
        >>> model_class.__name__
        'EfficientAd'
    """
    logger.info("Loading the model.")
    model_class: type[AnomalibModule] | None = None

    name = convert_snake_to_pascal_case(name).lower()
    for model in AnomalibModule.__subclasses__():
        if name == model.__name__.lower():
            model_class = model
    if model_class is None:
        logger.exception(f"Could not find the model {name}. Available models are {list_models()}")
        raise UnknownModelError

    return model_class


def get_model(model: DictConfig | str | dict | Namespace, *args, **kwdargs) -> AnomalibModule:
    """Get an anomaly detection model instance.

    This function instantiates an anomaly detection model based on the provided
    configuration or model name. It supports multiple ways of model specification
    including string names, dictionaries and OmegaConf configurations.

    Args:
        model (DictConfig | str | dict | Namespace): Model specification that can be:
            - A string with model name (e.g. ``"padim"``, ``"efficient_ad"``)
            - A dictionary with ``class_path`` and optional ``init_args``
            - An OmegaConf DictConfig with similar structure as dict
            - A Namespace object with similar structure as dict
        *args: Variable length argument list passed to model initialization.
        **kwdargs: Arbitrary keyword arguments passed to model initialization.

    Returns:
        AnomalibModule: Instantiated anomaly detection model.

    Raises:
        TypeError: If ``model`` argument is of unsupported type.
        UnknownModelError: If specified model class cannot be found.

    Examples:
        Get model by name:

        >>> model = get_model("padim")
        >>> model = get_model("efficient_ad")
        >>> model = get_model("patchcore", input_size=(100, 100))

        Get model using dictionary config:

        >>> model = get_model({"class_path": "Padim"})
        >>> model = get_model(
        ...     {"class_path": "Patchcore"},
        ...     input_size=(100, 100)
        ... )
        >>> model = get_model({
        ...     "class_path": "Padim",
        ...     "init_args": {"input_size": (100, 100)}
        ... })

        Get model using fully qualified path:

        >>> model = get_model({
        ...     "class_path": "anomalib.models.Padim",
        ...     "init_args": {"input_size": (100, 100)}
        ... })
    """
    model_: AnomalibModule
    if isinstance(model, str):
        model_class_ = _get_model_class_by_name(model)
        model_ = model_class_(*args, **kwdargs)
    elif isinstance(model, DictConfig | Namespace | dict):
        if isinstance(model, dict):
            model = OmegaConf.create(model)
        try:
            if len(model.class_path.split(".")) > 1:
                # Security check: Only allow imports from whitelisted modules
                module_path = ".".join(model.class_path.split(".")[:-1])
                if module_path not in ALLOWED_MODULES:
                    logger.error(
                        f"Module import from '{module_path}' is not allowed. "
                        f"Only imports from {ALLOWED_MODULES} are permitted.",
                    )
                    msg = f"Module import from '{module_path}' is not allowed."
                    raise UnknownModelError(msg)

                # Use a whitelist approach to prevent arbitrary code execution
                # nosemgrep: python.lang.security.audit.non-literal-import.non-literal-import
                module = import_module(module_path)
            else:
                module = import_module("anomalib.models")
        except ModuleNotFoundError as exception:
            logger.exception(
                f"Could not find the module {model.class_path}. Available models are {list_models()}",
            )
            raise UnknownModelError from exception
        try:
            model_class = getattr(module, model.class_path.split(".")[-1])
            init_args = model.get("init_args", {})
            if len(kwdargs) > 0:
                init_args.update(kwdargs)
            model_ = model_class(*args, **init_args)
        except AttributeError as exception:
            logger.exception(
                f"Could not find the model {model.class_path}. Available models are {list_models()}",
            )
            raise UnknownModelError from exception
    else:
        logger.error(f"Unsupported type {type(model)} for model configuration.")
        raise TypeError
    return model_
