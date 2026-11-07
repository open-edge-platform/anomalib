# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Pre-processing helpers for models trained by Anomalib Studio."""

from anomalib.data.transforms import ExportableCenterCrop
from anomalib.pre_processing import PreProcessor
from torchvision.transforms.v2 import CenterCrop, Compose, Resize, Transform

_CENTER_CROP_TYPES = (CenterCrop, ExportableCenterCrop)


def without_center_crop(pre_processor: PreProcessor | None) -> PreProcessor | None:
    """Replace center cropping with a direct resize to the crop size.

    Some models (e.g. Dinomaly) resize the input and then center-crop it by default.
    Their anomaly maps then only cover the central region of the input image, while
    Anomalib Studio stretches the anomaly map over the whole image or video frame.
    As a result, the heatmap is misaligned everywhere except in the center.

    This helper keeps the input resolution the model expects (the crop size) but
    resizes the whole image to it instead of cropping, so the anomaly map covers the
    full frame. Pre-processors without a top-level center crop are returned unchanged.

    Args:
        pre_processor: The model's pre-processor.

    Returns:
        PreProcessor | None: A pre-processor without center cropping, or the original
            pre-processor if it does not center-crop.
    """
    transform = getattr(pre_processor, "transform", None)
    if not isinstance(pre_processor, PreProcessor) or not isinstance(transform, Compose):
        return pre_processor

    crops = [t for t in transform.transforms if isinstance(t, _CENTER_CROP_TYPES)]
    if not crops:
        return pre_processor
    crop_size = list(crops[-1].size)

    transforms: list[Transform] = []
    has_resize = False
    for t in transform.transforms:
        if isinstance(t, _CENTER_CROP_TYPES):
            continue
        if isinstance(t, Resize):
            transforms.append(Resize(crop_size, interpolation=t.interpolation, antialias=t.antialias))
            has_resize = True
        else:
            transforms.append(t)

    if not has_resize:
        transforms.insert(0, Resize(crop_size, antialias=True))

    return PreProcessor(transform=Compose(transforms))
