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
            accepted_version=None,
            app_version="1.2.3",
            deployment_type=DeploymentType.DOCKER,
            licenses=[
                LicenseReference(
                    name="Apache 2.0 License",
                    url="https://www.apache.org/licenses/LICENSE-2.0",
                    required_for="Docker and development deployments",
                ),
            ],
        ),
    )
    system_service.accept_licenses = AsyncMock(
        return_value=LicenseAcceptanceResponse(accepted=True, accepted_version="1.2.3"),
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
    assert response.json() == {"accepted": True, "accepted_version": "1.2.3"}
    fxt_system_service.accept_licenses.assert_awaited_once()
