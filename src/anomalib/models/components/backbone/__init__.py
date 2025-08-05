"""Common backbone models.

Example:
    >>> from anomalib.models.components.backbone import get_decoder
    >>> decoder = get_decoder()

See Also:
    - :func:`anomalib.models.components.backbone.de_resnet`:
        Decoder network implementation
"""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .resnet_decoder import get_decoder

__all__ = ["get_decoder"]
