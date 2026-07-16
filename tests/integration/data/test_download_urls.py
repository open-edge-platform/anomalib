# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Test that dataset and model download URLs defined in anomalib are reachable.

Uses a manually curated list of DownloadInfo objects imported from datamodules
and model lightning modules. When adding a new DownloadInfo to the codebase,
add a corresponding entry to ``ALL_DOWNLOAD_INFOS`` below.
"""

import time
import urllib.request
from urllib.error import HTTPError, URLError

import pytest

from anomalib.data.datamodules.depth.adam_3d import DOWNLOAD_INFO as ADAM3D_INFO
from anomalib.data.datamodules.depth.mvtec_3d import DOWNLOAD_INFO as MVTEC3D_INFO
from anomalib.data.datamodules.image.bmad import DOWNLOAD_INFO as BMAD_INFO
from anomalib.data.datamodules.image.btech import DOWNLOAD_INFO as BTECH_INFO
from anomalib.data.datamodules.image.kolektor import DOWNLOAD_INFO as KOLEKTOR_INFO
from anomalib.data.datamodules.image.mvtecad import DOWNLOAD_INFO as MVTECAD_INFO
from anomalib.data.datamodules.image.mvtecad2 import DOWNLOAD_INFO as MVTECAD2_INFO
from anomalib.data.datamodules.image.vad import DOWNLOAD_INFO as VAD_INFO
from anomalib.data.datamodules.image.visa import DOWNLOAD_INFO as VISA_INFO
from anomalib.data.datamodules.video.avenue import ANNOTATIONS_DOWNLOAD_INFO as AVENUE_ANNOTATIONS_INFO
from anomalib.data.datamodules.video.avenue import DATASET_DOWNLOAD_INFO as AVENUE_DATASET_INFO
from anomalib.data.datamodules.video.ucsd_ped import DOWNLOAD_INFO as UCSD_INFO
from anomalib.data.datasets.image.mvtec_loco import DOWNLOAD_INFO as MVTEC_LOCO_INFO
from anomalib.data.utils import DownloadInfo
from anomalib.models.image.cfm.components import POINTMAE_DOWNLOAD_INFO
from anomalib.models.image.draem.lightning_model import DTD_DOWNLOAD_INFO as DRAEM_DTD_INFO
from anomalib.models.image.dsr.lightning_model import WEIGHTS_DOWNLOAD_INFO as DSR_WEIGHTS_INFO
from anomalib.models.image.efficient_ad.lightning_model import IMAGENETTE_DOWNLOAD_INFO
from anomalib.models.image.efficient_ad.lightning_model import WEIGHTS_DOWNLOAD_INFO as EFFICIENTAD_WEIGHTS_INFO
from anomalib.models.image.glass.lightning_model import DTD_DOWNLOAD_INFO as GLASS_DTD_INFO

ALL_DOWNLOAD_INFOS = [
    pytest.param(ADAM3D_INFO, id="adam_3d"),
    pytest.param(MVTEC3D_INFO, id="mvtec_3d"),
    pytest.param(BMAD_INFO, id="bmad"),
    pytest.param(BTECH_INFO, id="btech"),
    pytest.param(KOLEKTOR_INFO, id="kolektor"),
    pytest.param(MVTECAD_INFO, id="mvtecad"),
    pytest.param(MVTECAD2_INFO, id="mvtecad2"),
    pytest.param(VAD_INFO, id="vad"),
    pytest.param(VISA_INFO, id="visa"),
    pytest.param(AVENUE_DATASET_INFO, id="avenue_dataset"),
    pytest.param(AVENUE_ANNOTATIONS_INFO, id="avenue_annotations"),
    pytest.param(UCSD_INFO, id="ucsd_ped"),
    pytest.param(MVTEC_LOCO_INFO, id="mvtec_loco"),
    pytest.param(POINTMAE_DOWNLOAD_INFO, id="pointmae_weights"),
    pytest.param(DRAEM_DTD_INFO, id="draem_dtd"),
    pytest.param(DSR_WEIGHTS_INFO, id="dsr_weights"),
    pytest.param(IMAGENETTE_DOWNLOAD_INFO, id="efficientad_imagenette"),
    pytest.param(EFFICIENTAD_WEIGHTS_INFO, id="efficientad_weights"),
    pytest.param(GLASS_DTD_INFO, id="glass_dtd"),
]


@pytest.mark.network
@pytest.mark.parametrize("download_info", ALL_DOWNLOAD_INFOS)
def test_download_url_reachable(download_info: DownloadInfo) -> None:
    """Verify that the download URL returns a successful HTTP status.

    Uses HEAD with a fallback to GET for servers that reject HEAD.
    Retries on transient 5xx responses so momentary server outages do not
    produce false failures; only genuine 4xx (dead link) responses fail.
    """
    url = download_info.url
    timeout = 15
    max_attempts = 3

    def request_status() -> int | None:
        last_status = None
        for method in ("HEAD", "GET"):
            req = urllib.request.Request(url, method=method)  # noqa: S310
            try:
                resp = urllib.request.urlopen(req, timeout=timeout)  # nosemgrep # noqa: S310
            except HTTPError as exc:
                last_status = exc.code
            except URLError:
                if method == "GET":
                    return last_status
            else:
                status = resp.status
                resp.close()
                return status
        return last_status

    status = None
    for attempt in range(max_attempts):
        status = request_status()
        if status is not None and status < 500:
            break
        if attempt < max_attempts - 1:
            time.sleep(2 * (attempt + 1))

    if status is None:
        pytest.skip(f"Download URL for '{download_info.name}' is unreachable (network/transport error): {url}")
    if status >= 500:
        pytest.skip(f"Download URL for '{download_info.name}' returned transient HTTP {status} (server outage): {url}")

    assert status < 400, f"Download URL for '{download_info.name}' returned HTTP {status}: {url}"
