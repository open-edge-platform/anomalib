# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for video conversion utilities."""

from pathlib import Path

import cv2
import pytest
from pytest_mock import MockerFixture

from anomalib.data.utils.video import convert_video


@pytest.mark.parametrize("fps", [7.5, 24.0])
def test_convert_video_preserves_frame_rate(tmp_path: Path, mocker: MockerFixture, fps: float) -> None:
    """Pass the source frame rate to the output writer without truncation."""
    input_path = tmp_path / "input.avi"
    output_path = tmp_path / "output.avi"
    properties = {
        cv2.CAP_PROP_FRAME_WIDTH: 32.0,
        cv2.CAP_PROP_FRAME_HEIGHT: 24.0,
        cv2.CAP_PROP_FPS: fps,
    }
    capture = mocker.patch.object(cv2, "VideoCapture").return_value
    capture.get.side_effect = properties.__getitem__
    capture.read.return_value = False, None
    writer = mocker.patch.object(cv2, "VideoWriter")

    convert_video(input_path, output_path, codec="XVID")

    writer.assert_called_once_with(str(output_path), cv2.VideoWriter_fourcc(*"XVID"), fps, (32, 24))
