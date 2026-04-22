# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import status

from api.dependencies.dependencies import get_system_service
from main import app
from pydantic_models.system import DeploymentType, LicenseAcceptanceResponse, LicenseReference, LicenseStatus
from services import SystemService


@pytest.fixture
def fxt_system_service() -> MagicMock:
    system_service = MagicMock(spec=SystemService)
    system_service.get_license_status = AsyncMock(
        return_value=LicenseStatus(
            accepted=False,
            app_version="1.2.3",
            deployment_type=DeploymentType.WIN_APP,
            license=LicenseReference(
                name="Intel Simplified Software License",
                url="https://www.intel.com/content/www/us/en/content-details/749362/intel-simplified-software-license-version-october-2022.html",
                required_for="Windows application",
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
