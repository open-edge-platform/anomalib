# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import base64
import io

import numpy as np
from PIL import Image

from pydantic_models import PredictionLabel, PredictionResponse
from utils.visualization import Visualizer


def _rgba_png_base64(rgba: np.ndarray) -> str:
    """Encode an RGBA uint8 image as base64 PNG (matches ModelService output format)."""
    if rgba.dtype != np.uint8:
        raise ValueError("Expected uint8 RGBA image")
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError("Expected (H, W, 4) RGBA image")

    im = Image.fromarray(rgba)
    with io.BytesIO() as buf:
        im.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")


def test_overlay_anomaly_heatmap_uses_png_alpha() -> None:
    """Visualizer should composite the heatmap using the PNG's alpha channel scaled by the alpha parameter."""
    # Base image in RGB format (as used by the visualizer)
    base_rgb = np.full((2, 2, 3), (30, 20, 10), dtype=np.uint8)

    # Synthetic RGBA overlay; OpenCV will decode PNG as BGRA, then convert to RGB
    overlay_rgba = np.zeros((2, 2, 4), dtype=np.uint8)
    overlay_rgba[0, 0] = [255, 0, 0, 128]  # red @ 50%
    overlay_rgba[0, 1] = [0, 255, 0, 255]  # green @ 100%
    overlay_rgba[1, 0] = [0, 0, 255, 0]  # blue @ 0% (fully transparent)
    overlay_rgba[1, 1] = [255, 255, 255, 64]  # white @ 25%

    pred = PredictionResponse(
        anomaly_map=_rgba_png_base64(overlay_rgba),
        label=PredictionLabel.NORMAL,
        score=0.1,
    )

    # Use alpha=1.0 to have the PNG's alpha used directly
    out = Visualizer.overlay_anomaly_heatmap(base_rgb, pred, alpha=1.0)

    # Expected: RGB composite using PNG's alpha channel
    overlay_rgb = overlay_rgba[:, :, :3].astype(np.float32)
    a = (overlay_rgba[:, :, 3].astype(np.float32) / 255.0)[:, :, None]  # (H,W,1)
    expected = (base_rgb.astype(np.float32) * (1.0 - a) + overlay_rgb * a).astype(np.uint8)

    assert out.shape == expected.shape
    assert np.array_equal(out, expected)
