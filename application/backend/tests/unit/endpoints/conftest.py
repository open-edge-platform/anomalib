# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture
def fxt_client():
    return TestClient(app, raise_server_exceptions=False)
