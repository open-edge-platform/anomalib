# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Test torch inference entrypoint script."""

import sys
from collections.abc import Callable
from importlib.util import find_spec
from pathlib import Path

import pytest

from anomalib.models import Padim

sys.path.append("tools/inference")


class TestTorchInferenceEntrypoint:
    """This tests whether the entrypoints run without errors without quantitative measure of the outputs."""

    @pytest.fixture()
    @staticmethod
    def get_functions() -> tuple[Callable, Callable]:
        """Get functions from torch_inference.py."""
        if find_spec("torch_inference") is not None:
            from tools.inference.torch_inference import get_parser, infer
        else:
            msg = "Unable to import torch_inference.py for testing"
            raise ImportError(msg)
        return get_parser, infer

    @staticmethod
    def test_torch_inference(
        get_functions: tuple[Callable, Callable],
        project_path: Path,
        ckpt_path: Callable[[str], Path],
        get_dummy_inference_image: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test torch_inference.py."""
        # Set TRUST_REMOTE_CODE environment variable for the test
        monkeypatch.setenv("TRUST_REMOTE_CODE", "1")

        checkpoint_path = ckpt_path("Padim")
        get_parser, infer = get_functions
        model = Padim.load_from_checkpoint(checkpoint_path)
        model.to_torch(
            export_root=checkpoint_path.parent.parent.parent,
        )
        arguments = get_parser().parse_args(
            [
                "--weights",
                str(checkpoint_path.parent.parent) + "/torch/model.pt",
                "--input",
                get_dummy_inference_image,
                "--output",
                str(project_path) + "/output.png",
            ],
        )
        infer(arguments)
