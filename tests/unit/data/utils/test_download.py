# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for download utilities, focusing on SSL handling (gh-3477)."""

import ssl
from unittest.mock import patch

import pytest

from anomalib.data.utils.download import DownloadInfo, _ssl_context, download_and_extract


class TestSSLContext:
    """Tests for the _ssl_context helper."""

    @staticmethod
    def test_default_does_not_change_ssl() -> None:
        """Without env var, the default SSL context should remain unchanged."""
        original = ssl._create_default_https_context  # noqa: SLF001
        with _ssl_context():
            assert ssl._create_default_https_context is original  # noqa: SLF001
        assert ssl._create_default_https_context is original  # noqa: SLF001

    @staticmethod
    def test_no_verify_disables_ssl(monkeypatch: pytest.MonkeyPatch) -> None:
        """When ANOMALIB_NO_VERIFY_SSL=1, verification should be disabled."""
        monkeypatch.setenv("ANOMALIB_NO_VERIFY_SSL", "1")
        original = ssl._create_default_https_context  # noqa: SLF001
        with _ssl_context():
            assert ssl._create_default_https_context is ssl._create_unverified_context  # noqa: SLF001, S323
        # Restored after exit
        assert ssl._create_default_https_context is original  # noqa: SLF001

    @staticmethod
    def test_no_verify_true_string(monkeypatch: pytest.MonkeyPatch) -> None:
        """'true' (case-insensitive) should also disable verification."""
        monkeypatch.setenv("ANOMALIB_NO_VERIFY_SSL", "True")
        with _ssl_context():
            assert ssl._create_default_https_context is ssl._create_unverified_context  # noqa: SLF001, S323

    @staticmethod
    def test_no_verify_false_keeps_ssl(monkeypatch: pytest.MonkeyPatch) -> None:
        """Arbitrary values should not disable verification."""
        monkeypatch.setenv("ANOMALIB_NO_VERIFY_SSL", "false")
        original = ssl._create_default_https_context  # noqa: SLF001
        with _ssl_context():
            assert ssl._create_default_https_context is original  # noqa: SLF001

    @staticmethod
    def test_context_restores_on_exception(monkeypatch: pytest.MonkeyPatch) -> None:
        """SSL context must be restored even if the body raises."""
        monkeypatch.setenv("ANOMALIB_NO_VERIFY_SSL", "1")
        original = ssl._create_default_https_context  # noqa: SLF001
        msg = "boom"
        with pytest.raises(ValueError, match=msg), _ssl_context():
            raise ValueError(msg)
        assert ssl._create_default_https_context is original  # noqa: SLF001


class TestDownloadSSLError:
    """Tests for SSL error handling in download_and_extract."""

    @staticmethod
    def test_ssl_error_gives_actionable_message(tmp_path: pytest.TempPathFactory) -> None:
        """An SSLCertVerificationError should be wrapped with guidance."""
        info = DownloadInfo(name="test", url="https://example.com/data.tar.gz", hashsum="abc123")
        ssl_err = ssl.SSLCertVerificationError("certificate verify failed")

        with (
            patch("anomalib.data.utils.download.urlretrieve", side_effect=ssl_err),
            pytest.raises(RuntimeError, match="ANOMALIB_NO_VERIFY_SSL"),
        ):
            download_and_extract(root=tmp_path, info=info)

    @staticmethod
    def test_invalid_scheme_raises(tmp_path: pytest.TempPathFactory) -> None:
        """Non-http(s) URLs should raise RuntimeError."""
        info = DownloadInfo(name="test", url="ftp://example.com/data.tar.gz", hashsum="abc")
        with pytest.raises(RuntimeError, match="Invalid URL"):
            download_and_extract(root=tmp_path, info=info)
