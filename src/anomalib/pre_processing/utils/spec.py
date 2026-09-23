# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Serialize pre-processing transforms as safe, plain-data specifications."""

from __future__ import annotations

import importlib
import inspect
from enum import Enum
from typing import TypeAlias, cast, get_type_hints

from torchvision.transforms.v2 import Transform
from typing_extensions import TypedDict

ALLOWED_TRANSFORM_ROOTS = ("torchvision.transforms.", "anomalib.")

SpecValue: TypeAlias = str | int | float | bool | list["SpecValue"] | dict[str, "SpecValue"] | None
ConstructorValue: TypeAlias = SpecValue | Transform | Enum | list["ConstructorValue"] | dict[str, "ConstructorValue"]


class TransformSpec(TypedDict):
    """Plain-data representation of a transform."""

    class_path: str
    init_args: dict[str, SpecValue]


def transform_to_spec(transform: Transform | None) -> TransformSpec | None:
    """Convert a torchvision or anomalib transform to a plain-data spec.

    Args:
        transform: Transform to serialize, or ``None``.

    Returns:
        Nested transform specification, or ``None``.

    Raises:
        ValueError: If transform is outside the supported namespaces or its
            constructor arguments cannot be represented as plain data.
    """
    if transform is None:
        return None

    class_path = _class_path(type(transform))
    signature = inspect.signature(type(transform).__init__)
    init_args: dict[str, SpecValue] = {}
    for name, parameter in signature.parameters.items():
        if name == "self" or parameter.kind in {parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD}:
            continue
        if not hasattr(transform, name):
            msg = f"Cannot serialize {class_path}: missing constructor attribute {name!r}"
            raise ValueError(msg)
        init_args[name] = _to_plain_data(getattr(transform, name), class_path)

    return {"class_path": class_path, "init_args": init_args}


def spec_to_transform(spec: TransformSpec | None) -> Transform | None:
    """Construct a transform from a plain-data specification.

    Args:
        spec: Nested transform specification, or ``None``.

    Returns:
        Reconstructed transform, or ``None``.

    Raises:
        ValueError: If the specification is malformed or outside the supported
            namespaces.
    """
    if spec is None:
        return None
    if not isinstance(spec, dict) or set(spec) != {"class_path", "init_args"}:
        msg = "Transform spec must contain only class_path and init_args"
        raise TypeError(msg)

    class_path = spec["class_path"]
    if not isinstance(class_path, str) or not _is_allowed(class_path):
        msg = f"Unsupported transform class path: {class_path!r}"
        raise ValueError(msg)
    init_args = spec["init_args"]
    if not isinstance(init_args, dict):
        msg = f"Transform init_args must be a dictionary: {class_path}"
        raise TypeError(msg)

    cls = _resolve_class(class_path)
    signature = inspect.signature(cls.__init__)
    hints = get_type_hints(cls.__init__)
    args: dict[str, ConstructorValue] = {}
    for name, value in init_args.items():
        if name not in signature.parameters:
            msg = f"Unknown constructor argument {name!r} for {class_path}"
            raise ValueError(msg)
        args[name] = _from_plain_data(value, cast("type[Enum] | None", hints.get(name)))
    try:
        return cls(**args)
    except (TypeError, ValueError) as exc:
        msg = f"Invalid transform spec for {class_path}: {exc}"
        raise ValueError(msg) from exc


def _class_path(cls: type) -> str:
    """Get a stable public class path for a supported transform."""
    module = cls.__module__
    if module.startswith("torchvision.transforms.v2."):
        return f"torchvision.transforms.v2.{cls.__name__}"
    path = f"{module}.{cls.__name__}"
    if not _is_allowed(path):
        msg = f"Unsupported transform class: {path}"
        raise ValueError(msg)
    return path


def _is_allowed(class_path: str) -> bool:
    return class_path.startswith(ALLOWED_TRANSFORM_ROOTS)


def _resolve_class(class_path: str) -> type[Transform]:
    module_name, _, class_name = class_path.rpartition(".")
    try:
        cls = getattr(importlib.import_module(module_name), class_name)
    except (ImportError, AttributeError) as exc:
        msg = f"Cannot resolve transform class: {class_path}"
        raise ValueError(msg) from exc
    if not inspect.isclass(cls) or not issubclass(cls, Transform):
        msg = f"Not a transform class: {class_path}"
        raise ValueError(msg)
    return cls


def _to_plain_data(value: object, class_path: str) -> SpecValue:
    if isinstance(value, Transform):
        return transform_to_spec(value)  # type: ignore[return-value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_to_plain_data(item, class_path) for item in value]
    msg = f"Unsupported argument value: {type(value).__name__}"
    raise ValueError(msg)


def _from_plain_data(value: SpecValue, annotation: type[Enum] | None = None) -> ConstructorValue:
    if isinstance(value, dict) and "class_path" in value:
        return spec_to_transform(cast("TransformSpec", value))
    if isinstance(value, dict):
        return {key: _from_plain_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_from_plain_data(item) for item in value]
    if annotation is not None and inspect.isclass(annotation) and issubclass(annotation, Enum):
        return annotation(value)
    return value
