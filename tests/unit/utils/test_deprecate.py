# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the deprecation utility function."""

import pytest

from anomalib.utils import deprecate


def test_deprecated_function_warns() -> None:
    """Test simple function deprecation warning."""

    @deprecate(since="1.0", remove="2.0", use="new_func")
    def old_func() -> str:
        return "hello"

    with pytest.warns(DeprecationWarning, match=r"(?=.*1\.0)(?=.*2\.0)(?=.*new_func)"):
        assert old_func() == "hello"


def test_deprecated_class_warns_on_init() -> None:
    """Test deprecated class emits warning on instantiation."""

    @deprecate(since="1.0", remove="2.0", use="NewClass")
    class OldClass:
        def __init__(self) -> None:
            self.val = 42

    with pytest.warns(DeprecationWarning, match=r"(?=.*1\.0)(?=.*2\.0)(?=.*NewClass)"):
        instance = OldClass()
    assert instance.val == 42


def test_deprecated_function_arg_warns_with_none_replacement() -> None:
    """Test deprecated argument triggers warning when replacement is None."""

    @deprecate(args={"old_param": None}, since="1.0", remove="2.0")
    def func(new_param: int | None = None, old_param: int | None = None) -> int:
        return new_param if new_param is not None else old_param

    with pytest.warns(DeprecationWarning, match=r"(?=.*old_param)(?=.*1\.0)(?=.*2\.0)"):
        assert func(old_param=10) == 10


def test_deprecated_function_arg_warns() -> None:
    """Test deprecated argument triggers warning with correct match."""

    @deprecate(args={"old_param": "new_param"}, since="1.0", remove="2.0")
    def func(new_param: int | None = None, old_param: int | None = None) -> int:
        return new_param if new_param is not None else old_param

    with pytest.warns(DeprecationWarning, match=r"(?=.*old_param)(?=.*new_param)(?=.*1\.0)(?=.*2\.0)"):
        assert func(old_param=10) == 10


def test_deprecated_multiple_args_warns() -> None:
    """Test multiple deprecated arguments trigger appropriate warnings."""

    @deprecate(args={"old1": "new1", "old2": "new2"}, since="1.0")
    def func(new1=None, new2=None, old1=None, old2=None):  # noqa: ANN001, ANN202
        return new1 or old1, new2 or old2

    with pytest.warns(DeprecationWarning, match=r"(?=.*old1)(?=.*new1)(?=.*1\.0)"):
        func(old1="a")

    with pytest.warns(DeprecationWarning, match=r"(?=.*old2)(?=.*new2)(?=.*1\.0)"):
        func(old2="b")


def test_deprecated_class_arg_warns() -> None:
    """Test deprecated constructor argument triggers warning."""

    @deprecate(args={"legacy_arg": "modern_arg"}, since="1.0", remove="2.0")
    class MyClass:
        def __init__(self, modern_arg=None, legacy_arg=None) -> None:  # noqa: ANN001
            self.value = modern_arg if modern_arg is not None else legacy_arg

    with pytest.warns(DeprecationWarning, match=r"(?=.*legacy_arg)(?=.*modern_arg)(?=.*1\.0)(?=.*2\.0)"):
        obj = MyClass(legacy_arg="deprecated")
    assert obj.value == "deprecated"
