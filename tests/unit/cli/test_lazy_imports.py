# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for lazy import behavior and fast CLI startup."""

import subprocess
import sys
from pathlib import Path

import pytest


def test_cli_import_does_not_load_torch() -> None:
    """Verify importing AnomalibCLI does not eagerly import torch or anomalib.models."""
    code = (
        "import sys\n"
        "from anomalib.cli import AnomalibCLI\n"
        "assert 'torch' not in sys.modules, 'torch was eagerly imported!'\n"
        "assert 'anomalib.models' not in sys.modules, 'anomalib.models was eagerly imported!'\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)  # noqa: S603
    assert result.returncode == 0, f"Import test failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"


def test_cli_help_does_not_load_torch() -> None:
    """Verify running root CLI help does not eagerly import torch or vision backbones."""
    code = (
        "import sys\n"
        "from anomalib.cli.cli import AnomalibCLI\n"
        "try:\n"
        "    AnomalibCLI(['--help'])\n"
        "except SystemExit:\n"
        "    pass\n"
        "assert 'torch' not in sys.modules, 'torch was imported on --help!'\n"
        "assert 'anomalib.models' not in sys.modules, 'anomalib.models was imported on --help!'\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)  # noqa: S603
    assert result.returncode == 0, f"Help test failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"


def test_subcommand_parser_lazy_population() -> None:
    """Verify subcommand parsers populate arguments on demand without breaking external access."""
    from anomalib.cli.cli import AnomalibCLI

    cli = AnomalibCLI(args=["install"], run=False)
    # Fit parser was not active, but accessing it through the lazy dict must populate it
    fit_parser = cli.subcommand_parsers["fit"]
    action_dests = {action.dest for action in fit_parser._actions}  # noqa: SLF001
    assert "model" in action_dests
    assert "data" in action_dests

    # Benchmark parser was not active, but accessing it through the lazy dict must populate it
    benchmark_parser = cli.subcommand_parsers["benchmark"]
    benchmark_dests = {action.dest for action in benchmark_parser._actions}  # noqa: SLF001
    assert "config" in benchmark_dests


def test_subcommand_from_config_file_populates_parser(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that specifying a subcommand in a config file populates the subparser."""
    from jsonargparse import ArgumentParser

    from anomalib.cli.cli import AnomalibCLI, _extract_subcommand

    all_known = {"fit", "validate", "test", "train", "predict", "export", "install"}

    # Test 1: explicit 'subcommand: fit' key
    cfg_file1 = tmp_path / "config1.yaml"
    cfg_file1.write_text("subcommand: fit\n")

    assert _extract_subcommand(["--config", str(cfg_file1)], all_known) == "fit"

    def _noop(*_args: object, **_kwargs: object) -> None:
        return

    def _fake_parse_args(*_args: object, **_kwargs: object) -> dict[str, str]:
        return {"subcommand": "fit"}

    monkeypatch.setattr(AnomalibCLI, "before_instantiate_classes", _noop)
    monkeypatch.setattr(AnomalibCLI, "instantiate_classes", _noop)
    monkeypatch.setattr(ArgumentParser, "parse_args", _fake_parse_args)

    cli1 = AnomalibCLI(args=["--config", str(cfg_file1)], run=False)
    fit_parser1 = cli1.subcommand_parsers["fit"]
    dests1 = {action.dest for action in fit_parser1._actions}  # noqa: SLF001
    assert "model" in dests1
    assert "data" in dests1

    # Test 2: top-level subcommand mapping (e.g. 'fit: {...}')
    cfg_file2 = tmp_path / "config2.yaml"
    cfg_file2.write_text("fit:\n  default_root_dir: ./results\n")

    assert _extract_subcommand(["-c", str(cfg_file2)], all_known) == "fit"

    cli2 = AnomalibCLI(args=["--config", str(cfg_file2)], run=False)
    fit_parser2 = cli2.subcommand_parsers["fit"]
    dests2 = {action.dest for action in fit_parser2._actions}  # noqa: SLF001
    assert "model" in dests2
    assert "data" in dests2


def test_install_subcommand_skips_class_instantiation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that anomalib install does not instantiate classes or trigger _LazyImports.load()."""
    from anomalib.cli.cli import AnomalibCLI, _LazyImports

    called: dict[str, bool] = {"before": False, "instantiate": False, "load": False}

    def _track_before(*_args: object, **_kwargs: object) -> None:
        called["before"] = True

    def _track_instantiate(*_args: object, **_kwargs: object) -> None:
        called["instantiate"] = True

    def _track_load(*_args: object, **_kwargs: object) -> None:
        called["load"] = True

    monkeypatch.setattr(AnomalibCLI, "before_instantiate_classes", _track_before)
    monkeypatch.setattr(AnomalibCLI, "instantiate_classes", _track_instantiate)
    monkeypatch.setattr(_LazyImports, "load", classmethod(_track_load))

    cli = AnomalibCLI(args=["install"], run=False)
    assert cli.subcommand == "install"
    assert not called["before"], "before_instantiate_classes was called for install!"
    assert not called["instantiate"], "instantiate_classes was called for install!"
    assert not called["load"], "_LazyImports.load() was called for install!"
