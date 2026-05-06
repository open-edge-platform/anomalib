# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Self

import shortuuid
from pydantic_core import core_schema

if TYPE_CHECKING:
    from pydantic import GetCoreSchemaHandler
    from pydantic_core import CoreSchema

_SHORTUUID_RE = re.compile(r"^[23456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz]{22}$")


class ShortUUID(str):
    """ShortUUID.

    Convenience wrapper for shortuuid.uuid().

    Example:
        >>> ShortUUID()
        'iAAFNtgPsJEX2TnLJszCFh'
        >>> ShortUUID.generate()
        'JMuZHsGgoQZwUtEVTi5sea'
        >>> ShortUUID("JMuZHsGgoQZwUtEVTi5sea")
        'JMuZHsGgoQZwUtEVTi5sea'
        >>> ShortUUID("invalid")
        Traceback (most recent call last):
        ...
        ValueError: Invalid ShortUUID: 'invalid'

    """

    def __new__(cls, value: str = "") -> Self:
        if value and not _SHORTUUID_RE.match(value):
            raise ValueError(f"Invalid ShortUUID: {value!r}")
        if not value:
            return cls(shortuuid.uuid())
        return super().__new__(cls, value)

    @classmethod
    def generate(cls) -> ShortUUID:
        """Generate a new random ShortUUID.

        Example:
            >>> ShortUUID.generate()
            'JMuZHsGgoQZwUtEVTi5sea'

        Returns:
            A new random ShortUUID.
        """
        return cls(shortuuid.uuid())

    @classmethod
    def _pydantic_validate(cls, value: object) -> ShortUUID:
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(value)
        raise ValueError(f"string required, got {type(value).__name__}")

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: type, handler: GetCoreSchemaHandler) -> CoreSchema:  # noqa: PLW3201
        return core_schema.chain_schema(
            [
                core_schema.str_schema(),
                core_schema.no_info_plain_validator_function(cls._pydantic_validate),
            ],
            serialization=core_schema.to_string_ser_schema(),
        )
