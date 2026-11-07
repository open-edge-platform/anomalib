# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Use the operating system certificate store for outbound HTTPS requests.

Python's HTTP clients (``requests``, ``httpx``, ``urllib``) verify TLS certificates
against the ``certifi`` bundle by default. That bundle does not include certificates
installed in the OS trust store, such as the root certificate of a corporate
TLS-inspecting proxy. On those machines, downloading pretrained weights
(e.g. from the Hugging Face hub) fails with ``CERTIFICATE_VERIFY_FAILED`` even though
the same URL opens fine in a browser.

``truststore`` makes the standard library ``ssl`` module verify against the native
trust store (Windows certificate store, macOS keychain, or the system OpenSSL
bundle on Linux), the same approach used by ``pip``.
"""

import os

from loguru import logger

DISABLE_ENV_VAR = "ANOMALIB_STUDIO_DISABLE_SYSTEM_CERTS"

_injected = False


def use_system_certificate_store() -> bool:
    """Make outbound HTTPS requests verify certificates against the OS trust store.

    Must be called in every process that performs downloads, because
    multiprocessing workers started with ``spawn`` do not inherit the patched
    ``ssl`` module. Calling it more than once in a process is a no-op.

    Set the ``ANOMALIB_STUDIO_DISABLE_SYSTEM_CERTS`` environment variable to ``1``
    to keep the default ``certifi`` behaviour.

    Returns:
        bool: ``True`` if the system certificate store is in use, ``False`` otherwise.
    """
    global _injected  # noqa: PLW0603

    if _injected:
        return True

    if os.environ.get(DISABLE_ENV_VAR, "").strip().lower() in {"1", "true", "yes"}:
        logger.debug(f"System certificate store disabled via {DISABLE_ENV_VAR}")
        return False

    try:
        import truststore  # noqa: PLC0415

        truststore.inject_into_ssl()
    except Exception as e:
        logger.warning(f"Could not enable the system certificate store, falling back to certifi: {e}")
        return False

    _injected = True
    logger.debug("Using the system certificate store for HTTPS verification")
    return True
