# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import status

from api.dependencies.dependencies import get_system_service
from main import app
from pydantic_models.system import LicenseAcceptanceResponse, LicenseInfo, LicenseStatus
from services import SystemService


@pytest.fixture
def fxt_system_service() -> MagicMock:
    system_service = MagicMock(spec=SystemService)
    system_service.get_license_status = AsyncMock(
        return_value=LicenseStatus(
            accepted=False,
            app_version="1.2.3",
            is_desktop=True,
            license=LicenseInfo(
                distribution_license_name="Intel Simplified Software License",
                distribution_license_url="https://www.intel.com/content/www/us/en/content-details/749362/intel-simplified-software-license-version-october-2022.html",
                source_license_name="Apache 2.0 License",
                source_license_url="https://www.apache.org/licenses/LICENSE-2.0",
                third_party_notices_url="https://github.com/open-edge-platform/anomalib/blob/main/third-party-programs.txt",
            ),
        ),
    )
    system_service.accept_licenses = AsyncMock(
        return_value=LicenseAcceptanceResponse(accepted=True),
    )
    app.dependency_overrides[get_system_service] = lambda: system_service
    return system_service


def test_get_license_status(fxt_client, fxt_system_service):
    response = fxt_client.get("/api/system/license")

    assert response.status_code == status.HTTP_200_OK
    assert response.json()["accepted"] is False
    assert response.json()["app_version"] == "1.2.3"
    fxt_system_service.get_license_status.assert_awaited_once()


def test_accept_license(fxt_client, fxt_system_service):
    response = fxt_client.post("/api/system/license:accept")

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"accepted": True}
    fxt_system_service.accept_licenses.assert_awaited_once()
