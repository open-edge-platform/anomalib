# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for lazy import behavior and fast CLI startup."""

import subprocess
import sys


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
