# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import ssl
import sys
import types

import pytest

from utils import system_certs


@pytest.fixture(autouse=True)
def reset_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(system_certs, "_injected", False)
    monkeypatch.delenv(system_certs.DISABLE_ENV_VAR, raising=False)


def _fake_truststore(monkeypatch: pytest.MonkeyPatch, inject) -> types.ModuleType:
    module = types.ModuleType("truststore")
    module.inject_into_ssl = inject
    monkeypatch.setitem(sys.modules, "truststore", module)
    return module


class TestUseSystemCertificateStore:
    def test_injects_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = []
        _fake_truststore(monkeypatch, lambda: calls.append(1))

        assert system_certs.use_system_certificate_store() is True
        assert system_certs.use_system_certificate_store() is True
        assert calls == [1]

    @pytest.mark.parametrize("value", ["1", "true", "YES"])
    def test_can_be_disabled(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        calls = []
        _fake_truststore(monkeypatch, lambda: calls.append(1))
        monkeypatch.setenv(system_certs.DISABLE_ENV_VAR, value)

        assert system_certs.use_system_certificate_store() is False
        assert calls == []

    def test_failure_does_not_raise(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def broken() -> None:
            raise RuntimeError("boom")

        _fake_truststore(monkeypatch, broken)

        assert system_certs.use_system_certificate_store() is False
        assert system_certs._injected is False

    def test_patches_ssl_context(self) -> None:
        truststore = pytest.importorskip("truststore")
        original = ssl.SSLContext
        try:
            assert system_certs.use_system_certificate_store() is True
            assert ssl.SSLContext is truststore.SSLContext
        finally:
            truststore.extract_from_ssl()
            assert ssl.SSLContext is original
